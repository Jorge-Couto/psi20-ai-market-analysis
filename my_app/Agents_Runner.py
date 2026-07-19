from datetime import datetime
import pandas as pd
from pathlib import Path
from crewai import Agent, Task, LLM, Crew, Process
from crewai.tools import BaseTool, tool
from ddgs import DDGS
import os
from dotenv import load_dotenv
import time
import openai
from openai import AzureOpenAI

def keyword_relevance(title, body, company):
    text = (title + " " + body).lower()
    company_clean = company.lower().replace("_", " ")

    tokens = company_clean.split()

    # Strong match: full name
    if company_clean in text:
        return True

    # Medium match: at least 2 tokens present
    matches = sum(token in text for token in tokens)
    return matches >= 2

def llm_relevance_filter(title, body, company, llm):
    prompt = f"""
    Determine if the following news is relevant to {company}
    or its stock, operations, sector, or macro factors affecting it.

    Answer ONLY: YES or NO

    News:
    Title: {title}
    Content: {body}
    """

    response = llm(prompt)
    return "YES" in response.upper()

def fetch_news(company_query):
    current_year = str(datetime.now().year)
    queries = [
        # direct company
        company_query,
        f"{company_query} portugal",
        f"{company_query} market",

        # management / governance catalyst
        f"{company_query} executive leadership",

        # earnings / financial catalysts
        f"{company_query} earnings",
        f"{company_query} guidance",
        f"{company_query} dividend",

        # corporate actions
        f"{company_query} acquisition",
        f"{company_query} partnerships",
        f"{company_query} new projects",

        # regulation / legal
        f"{company_query} regulation",
        f"{company_query} lawsuit",

        # macro / rates
        "ecb interest rates",
        "eurozone inflation",
        "portugal bond yields",

        # PSI20 performance news
        "psi 20 index",
        "psi 20 performance",
        "portugal stock market news",

        # geopolitical
        "global tariffs europe",
        "war escalation and trade tensions",
        "ukraine war europe impact",
        "middle east conflict oil prices",
        "china europe trade tensions",
        "eu tariffs policy",

        # operational / commodity
        f"{company_query} commodity prices",
        f"{company_query} supply chain",
    ]

    collected = []
    seen_titles = set()

    with DDGS() as ddgs:
        for q in queries:
            try:
                results = ddgs.news(q, max_results=20, timelimit="m")
                for r in results:
                    title = r.get("title", "")
                    date_str = r.get("date", "")
                    body = r.get("body", "")
                    if title in seen_titles:
                        continue

                    # 2 Step filtering to ensure news are related to the query
                    # Step 1: keyword filter
                    #if not keyword_relevance(title, body, company_query):
                        # Step 2: LLM fallback
                    #    if not llm_relevance_filter(title, body, company_query, llm):
                    #        continue
                    
                    seen_titles.add(title)
                    collected.append(
                        f"T: {title}\n"
                        f"D: {date_str}\n"
                        f"S: {body[:200]}...\n"
                    )

                    if len(collected) >= 20: break
            except Exception: continue
            if len(collected) >= 20: break

    return "\n---\n".join(collected)

def run_agents_for_company(company):
    # -----------------------
    # DEFINE LLM
    # -----------------------
    llm = LLM(
        model="", # Insert your model name here
        api_key=os.getenv("API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        api_version="2024-12-01-preview",
        # If you want to use your own model instead of the Azure-hosted one:
        # base_url=os.getenv("BASE_URL"),
    )

    # -----------------------
    # AGENTS
    # -----------------------
    news_analyzer = Agent(
        role="Energy News Analyst",

        goal="""
        Analyze recent news related to {topic} and extract the underlying events
        that may influence the company's stock price.

        For each news item:
        - Identify the core event (not the headline wording)
        - Classify the type of event (earnings, macro, regulation, etc.)
        - Explain the economic or financial mechanism through which the event
          could affect the company

        Additionally, incorporate the timing of the news:
        - Identify whether the event is very recent, recent, or older compared to {current_date}, which is the curent date
        - Note that recency affects how strongly the market is likely to react,
          but DO NOT assign final impact or sentiment here

        Focus on:
        - material developments (not generic commentary)
        - whether the event has ongoing relevance or is likely already priced in
        """,

        backstory="""
        You are a senior equity research analyst specializing in European markets.

        Your role is to convert raw news into structured, investment-relevant events.

        You distinguish between:
        - structural vs short-lived events
        - company-specific vs macro-driven developments
        - new information vs already digested news

        You are highly sensitive to timing:
        markets react differently to fresh information compared to older news.

        However, you do NOT assign sentiment or final impact.
        Your role is to prepare clean, structured inputs for downstream analysis.
        """,

        llm=llm,
        verbose=True
    )

    news_deduplicator = Agent(
        role="News Deduplication Specialist",

        goal="""
        Identify when multiple news articles describe the same underlying
        real-world event, even if they differ in wording, timing, source,
        or narrative framing.

        Consolidate duplicate articles into a single unique event record
        while preserving the most complete and information-rich representation.

        Treat articles as duplicates when they refer to the same:
        - corporate announcement
        - earnings release
        - regulatory action
        - macroeconomic event
        - geopolitical development
        - commodity price movement
        - executive decision
        - merger, acquisition, or partnership
        """,

        backstory="""
        You are a specialist in event-level financial information processing.

        Your expertise lies in distinguishing between:
        - multiple outlets reporting the same event
        - follow-up commentary on an existing event
        - genuinely new developments that change the original story

        You do not deduplicate based on similar wording alone.
        Instead, you focus on whether the underlying economic or corporate
        event is fundamentally the same.

        You preserve timeline integrity by keeping separate events when:
        - a story evolves materially
        - new facts emerge
        - market expectations change
        - management updates prior guidance
        - regulators escalate or reverse decisions

        Your goal is to prevent repeated headlines from artificially
        amplifying the perceived importance of a single event.

        """,

        llm=llm,
        verbose=True
    )

    news_impact_analyst = Agent(
        role="Market Sentiment Analyst",

        goal="""
        Evaluate each news event and determine its time-adjusted impact
        on the company’s stock price and investor sentiment.

        Your analysis must explicitly incorporate BOTH:
        1. intrinsic importance of the event
        2. recency of the event (time decay) compared to today's date, that is {current_date}

        For each event:

        Step 1 — Assess intrinsic impact (ignoring time):
        - How important is the event fundamentally?
        - Does it affect earnings, risk, growth, or sentiment?

        Step 2 — Apply recency weighting:
        Classify the event based on its date compared to {current_date}, that is today:
        - Immediate (0–3 days): full impact
        - Recent (4–10 days): slightly reduced impact
        - Fading (11–30 days): significantly reduced impact
        - Old (>30 days): minimal or already priced in

        Step 3 — Produce a final time-adjusted evaluation:
        Determine:
        - Direction of impact (Positive / Negative / Neutral)
        - Impact strength AFTER time adjustment (Low / Medium / High)
        - Time horizon of relevance (Short / Medium / Fading)
        - Whether the event is already priced in (Yes / Partially / No)
        - Confidence level of the assessment (Low / Medium / High)

        Important rules:
        - A high-impact but old event should have reduced influence
        - A recent moderate event may outweigh an older major event
        - Do NOT ignore older events, but discount their importance appropriately
        - Always explain how recency affected your final judgment
        """,

        backstory="""
        You are a senior market strategist specializing in event-driven trading,
        investor psychology, and short-term price formation.

        You understand that markets react not only to the importance of news,
        but also to WHEN that information becomes available.

        Your expertise lies in combining:
        - event materiality
        - timing and information flow
        - market expectations and pricing dynamics

        You distinguish between:
        - fresh catalysts that can still move prices
        - partially digested news with reduced impact
        - fully priced-in events with little remaining influence

        You think like institutional investors and hedge funds:
        the key question is not just "is this important?"
        but "is this STILL actionable?"

        Your analysis reflects how real markets discount older information
        and prioritize new signals.
        """,

        llm=llm,
        verbose=True
    )

    forecast_analyst = Agent(
        role="Quantitative Forecast and Risk Signal Analyst",

        goal="""
        Analyze the provided 5-business-day stock forecast data and translate
        the model outputs into actionable, risk-adjusted market expectations.

        The forecast originate from LSTM models with Monte Carlo dropout confidence intervals.

        Prioritize interpretation of:
        - expected return trend over the full 5-day horizon
        - persistence and acceleration of directional momentum
        - consistency between return path and forecasted price path
        - widening or tightening uncertainty bands
        - divergence between stable mean path and large uncertainty
        - volatility regime implied by confidence intervals
        - downside tail risk using 5D VaR and CVaR
        - probability of reversal, exhaustion, or breakout

        Determine whether the forecast implies:
        - sustained bullish continuation
        - sustained bearish continuation
        - low-conviction sideways consolidation
        - asymmetric downside risk
        - high-conviction trend persistence
        - fragile momentum vulnerable to uncertainty expansion
        """,

        backstory="""
        You are a senior quantitative financial analyst specialized in interpreting
        probabilistic stock forecasting systems for tactical investment decisions.

        Your expertise spans:
        - LSTM trend modeling
        - Monte Carlo scenario-based forecasting
        - LSTM neural forecasts with Monte Carlo dropout uncertainty
        - tail-risk interpretation using VaR and CVaR
        - confidence band regime analysis

        You do not focus on raw point forecasts alone.
        Your role is to convert forecast paths, confidence structures,
        and tail-risk metrics into decision-relevant market narratives.

        You distinguish between:
        - strong directional forecasts with controlled uncertainty
        - visually stable mean paths with hidden wide risk bands
        - statistically weak forecasts despite smooth trends
        - trend continuation with deteriorating downside asymmetry
        - low-volatility drift versus unstable forecast expansion

        Because LSTM MC-dropout forecasts may preserve smooth means 
        with wider uncertainty, your primary focus is:
        - direction of the mean return path
        - momentum persistence
        - confidence interval expansion
        - downside tail risk severity
        - practical actionability under uncertainty

        Your output should emphasize trend reliability,
        uncertainty-adjusted conviction, and portfolio usefulness.
        """,

        llm=llm,
        verbose=True
    )

    decision_agent = Agent(
        role="Risk-Aware Multi-Factor Investment Strategist",

        goal="""
        Integrate the probabilistic forecast signal with the time-adjusted
        impact of recent news events to produce a final short-term
        investment outlook for {topic}. Today's date is {current_date}.

        Your objective is to synthesize:
        - forecasted price direction and momentum
        - trend persistence and stability
        - uncertainty reflected in confidence intervals
        - downside risk (VaR / CVaR where relevant)
        - event-driven sentiment and catalyst strength
        - timing and recency of news (time decay)
        - macro and sector spillover effects

        Core principles:

        1. Forecast interpretation:
          - Prioritize trend direction and momentum over absolute predicted values
          - Evaluate whether the forecast path is stable, noisy, or fragile
          - Treat wide confidence intervals as reduced conviction signals

        2. News integration:
          - Weigh events based on BOTH importance and recency
          - Recent events should carry more influence than older ones
          - Older events may already be priced in unless highly material

        3. Conflict resolution:
          If forecast and news signals disagree:
          - Favor news when:
            • the event is recent and high-impact
            • the event invalidates key assumptions behind the forecast
          - Favor the forecast when:
            • news is older, weak, or already priced in
            • the forecast shows stable, persistent momentum
          - Reduce conviction when signals are mixed or uncertain

        4. Risk awareness:
          - Incorporate downside asymmetry where relevant
          - Identify whether tail risk meaningfully affects the decision
          - Avoid strong directional calls under high uncertainty

        Produce a final actionable stance:
        - Bullish
        - Cautiously Bullish
        - Neutral
        - Cautiously Bearish
        - Bearish

        Include:
        - conviction level
        - expected tactical horizon
        - key supporting drivers
        - explicit downside risks and invalidation scenarios
        """,

        backstory="""
        You are a senior portfolio strategist responsible for converting
        probabilistic forecasts and event-driven intelligence into
        actionable investment decisions.

        You specialize in combining:
        - machine learning forecast outputs (LSTM with uncertainty)
        - confidence interval interpretation
        - downside risk awareness
        - time-sensitive news catalysts
        - macroeconomic and sector-level dynamics

        into a coherent investment thesis.

        You understand that:
        - smooth forecast trends can hide significant uncertainty
        - recent news drives market reactions more than older information
        - large events can override statistically stable forecasts
        - weak or stale news should not outweigh consistent trend signals

        You distinguish between:
        - strong directional signals with supporting catalysts
        - fragile forecasts exposed to event risk
        - high-uncertainty environments requiring caution
        - tactical opportunities driven by recent catalysts
        - situations where no clear edge exists

        You think like an institutional investor:
        the goal is to determine whether the balance of expected return,
        confidence, and risk justifies taking exposure.

        Your final judgment emphasizes:
        - direction
        - conviction adjusted for uncertainty
        - most relevant (and recent) catalysts
        - key risks that could invalidate the thesis
        """,

        llm=llm,
        verbose=True
    )
    
    # -----------------------
    # TASKS
    # -----------------------
    task_news_analysis = Task(
        description="""
        Analyze the following news articles related to {topic}.

        NEWS INPUT:
        {news}

        Each article includes a publication date. You MUST use this
        information to assess how recent the event is. Today's date is {current_date}.

        Your objective is to extract the underlying real-world events,
        not to interpret stock price impact.

        For each article:

        1. Identify the single most important underlying event
          (not the headline wording, but what actually happened)

        2. Extract and consider the publication date:
          - Determine whether the event is:
            • Very recent (0–3 days)
            • Recent (4–10 days)
            • Older (11–30 days)
          - Note that older events may already be partially priced in

        3. Explain the direct connection to {topic}

        4. Classify the type of event:
          - corporate (earnings, guidance, M&A, management)
          - regulatory (laws, policy, government action)
          - macro (rates, inflation, economic conditions)
          - geopolitical (war, trade tensions)
          - sector (industry-wide developments)
          - commodity (energy, raw materials, pricing)

        5. Determine scope of impact:
          - company-specific
          - sector-wide
          - macro/systemic

        6. Assess intrinsic materiality (IGNORE timing here):
          - low / medium / high

        Important rules:
        - Focus only on factual, event-driven information
        - Ignore opinion pieces and generic commentary
        - If multiple articles describe the same event, still extract
          the event independently (deduplication happens later)
        - Do NOT assign sentiment or final market impact
        """,

        expected_output="""
        A structured event list, one item per article.

        For each event include:
        - Article identifier
        - Event date
        - Recency classification
          (Very recent / Recent / Older)
        - Main event summary
        - Event category
          (corporate / regulatory / macro / geopolitical / sector / commodity)
        - Why it matters for {topic}
        - Scope of relevance
          (company-specific / sector-wide / macro)
        - Intrinsic materiality
          (low / medium / high)
        """,

        agent=news_analyzer
    )

    task_deduplicate = Task(
        description="""
        Review the structured event list produced for {topic}
        and identify events that refer to the same underlying
        real-world development.

        Merge events only when they describe the same
        economic, corporate, regulatory, geopolitical,
        or sector event.

        Treat articles as duplicates even if:
        - wording differs
        - publication times differ
        - sources frame the story differently
        - sentiment tone differs
        - one article includes more background context

        IMPORTANT:
        Do NOT merge events when:
        - materially new facts are introduced
        - the story evolves with new guidance
        - the market implications change
        - a follow-up event modifies the original thesis
        - new operational or regulatory decisions emerge

        For each duplicate cluster:
        1. keep the most complete and information-rich event
        2. preserve the normalized event summary
        3. discard only true duplicates
        4. retain event timeline integrity

        """,

        expected_output="""
        A structured list of unique events for {topic}.

        For each retained unique event include:
        - Canonical event summary
        - Representative source article
        - Duplicate articles removed
        - Reason duplicates were merged
        - Whether the event is new or part of an evolving story
        """,

        agent=news_deduplicator,
        context=[task_news_analysis]
    )

    task_news_impact = Task(
        description="""
        Evaluate the likely stock market reaction to each unique
        event identified for {topic}.

        Each event includes a recency classification
        (Very recent / Recent / Older). You MUST incorporate this
        into your analysis. Today is {current_date}.

        Your task is to determine the **time-adjusted market impact**
        of each event by combining:

        1. intrinsic importance of the event
        2. recency (time decay)

        Step 1 — Intrinsic assessment (ignore timing):
        - How important is the event fundamentally?
        - Does it affect earnings, risk, growth, or sentiment?

        Step 2 — Apply time decay:
        - Very recent (0–3 days): full impact
        - Recent (4–10 days): moderately reduced impact
        - Older (11–30 days): significantly reduced impact

        Step 3 — Final market interpretation:
        Determine how market participants are likely to react NOW,
        after accounting for both importance and timing.

        Your analysis must include:
        - direction of expected stock reaction
        - impact strength AFTER time adjustment
        - expected persistence within a short-term trading horizon
        - whether the event is already priced in
        - whether the reaction is sentiment-driven or fundamentally justified
        - confidence in the assessment

        Distinguish carefully between:
        - fresh catalysts vs fading information
        - temporary sentiment reactions vs durable repricing
        - macro spillover vs company-specific impact

        Important rules:
        - A high-impact but older event must be discounted
        - A recent moderate event may outweigh an older major one
        - Do NOT ignore older events, but reduce their influence appropriately
        - Always reflect how recency affected your final judgment
        """,

        expected_output="""
        A structured impact assessment for each unique event.

        For each event include:
        - Canonical event summary
        - Recency classification
          (Very recent / Recent / Older)
        - Sentiment direction
          (Positive / Negative / Neutral)
        - Impact strength (time-adjusted)
          (Low / Medium / High)
        - Expected persistence (within ~5 trading days)
          (Short-lived / Moderate / Persistent)
        - Pricing status
          (Already priced in / Partially priced in / New information)
        - Impact type
          (Sentiment-driven / Fundamentally justified / Mixed)
        - Confidence level
          (Low / Medium / High)
        - Brief reasoning (must reference BOTH importance and recency)
        """,

        agent=news_impact_analyst,
        context=[task_deduplicate]
    )

    task_forecast_analysis = Task(
        description="""
        Analyze the following 5-business-day stock forecast data for {topic}:

        FORECAST INPUT:
        {forecast_data_str}

        The forecast includes:
        - forecast dates
        - expected mean return path
        - lower and upper return confidence intervals
        - expected forecast price path
        - lower and upper price confidence intervals
        - 5D VaR (95%)
        - 5D CVaR (05%)

        The forecast is generated using an LSTM model with Monte Carlo dropout,
        meaning the mean forecast path may appear smooth while uncertainty bands
        remain wide. Therefore, prioritize the **mean return trend, momentum persistence,
        and uncertainty structure** over absolute point prices.

        Your task is to interpret the full 10-business-day forecast path,
        not just the first and last values.

        Evaluate:
        1. overall direction of the mean return path
        2. strength and persistence of momentum
        3. consistency between return trend and forecast price path
        4. widening or narrowing of return and price confidence intervals
        5. volatility implied by confidence band width
        6. likelihood of reversal, exhaustion, or sideways drift
        7. downside tail severity using 5D VaR and CVaR
        8. whether the signal is sufficiently reliable for a tactical decision

        Distinguish between:
        - strong smooth continuation with acceptable uncertainty
        - stable mean path with excessively wide dropout bands
        - weak drift with poor conviction
        - reversal-prone unstable forecasts
        - asymmetric downside despite positive trend
        """,

        expected_output="""
        A structured quantitative forecast interpretation for {topic}.

        Include:
        - Expected direction
          (Bullish / Bearish / Sideways)
        - Momentum strength
          (Weak / Moderate / Strong)
        - Forecast persistence
          (Stable / Fragile / Reversal risk)
        - Confidence interval behavior
          (Narrowing / Stable / Widening)
        - Implied uncertainty regime
          (Low / Medium / High)
        - Tail risk assessment
          (Low / Moderate / Severe downside risk)
        - Overall forecast confidence
          (Low / Medium / High)
        - Tactical actionability
          (Strong signal / Usable with caution / Weak signal)
        - Brief reasoning
        """,

        agent=forecast_analyst
    )

    task_final_evaluation = Task(
        description="""
        Using the deduplicated news events (and their time-adjusted market impact)
        together with the 5-business-day LSTM Monte Carlo forecast analysis,
        produce a final short-term investment outlook for {topic} stocks.
        Consider today's date as {current_date}.

        IMPORTANT CONTEXT:
        - Forecast horizon is strictly 5 trading days
        - Forecast data contains BOTH:
            • return dynamics (mean_return, lower_return, upper_return)
            • price path (forecast_price, confidence intervals)
        - Return dynamics are the PRIMARY signal for trend interpretation
        - Price series is SECONDARY and used only for interpretability
        - Monte Carlo dropout bands reflect forecast uncertainty
        - VaR and CVaR represent downside tail risk over the same horizon

        FORECAST INTERPRETATION RULES:

        1. Primary signal (MOST IMPORTANT):
          - Focus on mean_return trajectory across 5 days
          - Determine:
            • direction of expected returns
            • acceleration or deceleration of momentum
            • stability of return path

        2. Secondary signal:
          - Use forecast_price ONLY to contextualize magnitude
          - Do NOT base direction on price alone

        3. Uncertainty integration:
          - Wide confidence bands (returns or price) → lower conviction
          - Divergence between upper/lower return bounds → instability warning
          - Strong mean trend + wide bands = directional bias with low confidence

        4. News integration:
          - Adjust forecast interpretation using time-weighted news impact
          - Recent high-impact news can override weak return signals
          - Older news should be discounted unless highly material

        5. Risk integration:
          - High VaR/CVaR → structural downside risk
          - Use tail risk to reduce bullish conviction or increase bearish bias
          - Tail risk can override weak positive momentum

        DECISION HIERARCHY:

        Step 1 — Forecast signal (baseline):
        - Identify return-based direction and momentum (mean_return path)
        - Assess volatility from return dispersion and CI width

        Step 2 — News adjustment:
        - Apply time-adjusted sentiment and event impact
        - Determine whether news reinforces or contradicts forecast

        Step 3 — Risk adjustment:
        - Incorporate VaR/CVaR tail risk
        - Adjust conviction based on downside asymmetry

        CONFLICT RULES:
        - Forecast + news aligned → increase conviction
        - Forecast strong but news adverse → reduce conviction or invert
        - Mixed weak signals → Neutral or cautious stance
        - High tail risk → defensive bias regardless of direction

        OUTPUT RULES:
        - Do NOT overemphasize price levels
        - Focus on return dynamics and risk-adjusted direction
        - Do NOT include meta commentary or filler text
        - Maintain institutional investment report tone

        Output structure:
        ---
        Investment Outlook for {topic}:
        - Final Recommendation:
          [Bullish / Cautiously Bullish / Neutral / Cautiously Bearish / Bearish]

        - Conviction Level:
          [Low / Medium / High]

        - Key Forecast Insights:
            • Return-based trend direction:
            • Momentum (acceleration/deceleration):
            • Volatility regime (from return dispersion):
            • Forecast confidence (MC dropout + CI width):
            • Tail risk (5D VaR / CVaR):

        - Key News Insights:
            • Most impactful events:
            • Event-driven sentiment:
            • Time-adjusted relevance:

        - Integrated Analysis:
            • Forecast + news synthesis:
            • Signal alignment or conflict resolution:
            • Risk-adjusted reasoning:

        - Primary Risks:
            • Downside risks and invalidation scenarios
            • Tail risk and uncertainty drivers
        ---

        Write in concise institutional investment style.
        Focus strictly on decision-grade reasoning.
        """,

        expected_output="""
        A structured institutional-grade investment report fully integrating:
        - 5-day return-based LSTM Monte Carlo forecasts
        - time-adjusted news impact
        - tail risk (VaR/CVaR)
        """,

        agent=decision_agent,
        context=[task_news_impact, task_forecast_analysis]
    )
    
    # -----------------------
    # CREW AI
    # -----------------------
    crew = Crew(
        agents=[
            news_analyzer,
            news_deduplicator,
            news_impact_analyst,
            forecast_analyst,
            decision_agent
        ],

        tasks=[
            task_news_analysis,
            task_deduplicate,
            task_news_impact,
            task_forecast_analysis,
            task_final_evaluation
        ],

        process=Process.sequential,
        verbose=True
    )

    # -----------------------
    # FETCH NEWS
    # -----------------------
    news = fetch_news(company)

    # -----------------------
    # LOAD FORECAST
    # -----------------------
    forecast_df = pd.read_csv(f"forecasts/{company}_forecast.csv")
    forecast_data_str = forecast_df.to_string(index=False)

    # -----------------------
    # INPUTS
    # -----------------------
    inputs = {
        "topic": company,
        "news": news,
        "forecast_data_str": forecast_data_str,
        "current_date": datetime.now().strftime("%Y-%m-%d")
    }

    # -----------------------
    # RUN CREW
    # -----------------------
    result = crew.kickoff(inputs=inputs)

    # -----------------------
    # SAVE RESULT
    # -----------------------
    Path("results").mkdir(exist_ok=True)

    with open(f"results/{company}_results.txt", "w", encoding="utf-8") as f:
        f.write(str(result))

    return str(result)