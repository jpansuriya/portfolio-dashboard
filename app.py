import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf

from portfolio_backtesting import PortfolioBacktester, demonstrate_backtesting

# Page setup
st.set_page_config(page_title="My Portfolio Dashboard", layout="wide")
st.title("📊 My Portfolio Dashboard")
# Sidebar option
st.sidebar.header("Settings")
use_live = st.sidebar.checkbox("Use Live Prices (Yahoo Finance)", value=True)

# Load CSV
df = pd.read_csv("Portfolio.csv")
df.columns = df.columns.str.strip()

# --- Fetch live prices only if toggle is ON ---
if use_live:
    for i, row in df.iterrows():
        ticker = yf.Ticker(row["Ticker"])
        try:
            live_price = ticker.history(period="1d")["Close"].iloc[-1]
            df.at[i, "Current Price"] = live_price
        except Exception:
            st.warning(f"⚠️ Could not fetch price for {row['Ticker']}")


# Calculate extra columns
df["Value"] = df["Shares"] * df["Current Price"]
df["Unrealized P/L"] = (df["Current Price"] - df["Buy Price"]) * df["Shares"]
df["% Gain/Loss"] = ((df["Current Price"] - df["Buy Price"]) / df["Buy Price"]) * 100

# Show portfolio summary
st.subheader("Portfolio Summary")
st.dataframe(df)

# Total portfolio value
st.metric("Total Portfolio Value", f"${df['Value'].sum():,.2f}")

# Allocation by Ticker
fig1 = px.pie(df, values="Value", names="Ticker", title="Portfolio Allocation by Ticker")
st.plotly_chart(fig1, use_container_width=True)

# Allocation by Sector
fig2 = px.pie(df, values="Value", names="Sector", title="Portfolio Allocation by Sector")
st.plotly_chart(fig2, use_container_width=True)

# Top gainers/losers
st.subheader("Performance Ranking")
st.dataframe(df.sort_values(by="% Gain/Loss", ascending=False))

# --- Top gainers/losers chart with colors ---
st.subheader("Top Gainers & Losers")

ranked = df.sort_values(by="% Gain/Loss", ascending=False).copy()
ranked["% Gain/Loss"] = ranked["% Gain/Loss"].round(2)

colors = ["green" if val > 0 else "red" for val in ranked["% Gain/Loss"]]

fig3 = px.bar(
    ranked,
    x="Ticker",
    y="% Gain/Loss",
    title="Top Gainers & Losers (%)",
    text="% Gain/Loss"
)

fig3.update_traces(marker_color=colors, texttemplate="%{text}%", textposition="outside")
fig3.update_layout(yaxis_title="% Gain/Loss", xaxis_title="Ticker")
st.plotly_chart(fig3, use_container_width=True)

# --- Portfolio vs Benchmarks ---
st.subheader("📈 Portfolio vs Market Benchmarks")

# Sidebar: select time period
period = st.sidebar.selectbox(
    "Select Time Period",
    ["1mo", "3mo", "6mo", "1y", "ytd", "5y"],
    index=2  # default "6mo"
)

# Download benchmark data
benchmark_tickers = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC"}
benchmark_data = {}
for name, ticker in benchmark_tickers.items():
    hist = yf.download(ticker, period=period)["Close"]
    benchmark_data[name] = hist

# Build portfolio history (approximation)
hist_data = pd.DataFrame()
for i, row in df.iterrows():
    ticker = row["Ticker"]
    shares = row["Shares"]
    try:
        prices = yf.download(ticker, period=period)["Close"]
        hist_data[ticker] = prices * shares
    except Exception:
        continue

portfolio_history = hist_data.sum(axis=1)

# Normalize portfolio and benchmarks to start at the same value
comparison = pd.DataFrame({"Portfolio": portfolio_history})
comparison = comparison / comparison.iloc[0] * 100

for name, series in benchmark_data.items():
    comparison[name] = series / series.iloc[0] * 100

# Plot
fig4 = px.line(comparison, title="Portfolio vs S&P 500 and Nasdaq")
st.plotly_chart(fig4, use_container_width=True)


# --- PORTFOLIO BACKTESTING SECTION ---
st.subheader("🚀 Advanced Portfolio Backtesting")

# Create tabs for different backtesting features
tab1, tab2, tab3 = st.tabs(["📊 Performance Analysis", "🎲 Monte Carlo Simulation", "📈 Strategy Comparison"])

with tab1:
    st.markdown("### Historical Performance & Risk Metrics")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("**Backtesting Options:**")
        years_back = st.slider("Years of Historical Data", 1, 5, 2)
        rebalance_freq = st.selectbox("Rebalancing Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"])
        
        if st.button("🚀 Run Performance Analysis", type="primary"):
            with st.spinner("Analyzing portfolio performance..."):
                try:
                    # Initialize backtester
                    backtester = PortfolioBacktester(df)
                    
                    # Calculate start date
                    from datetime import datetime, timedelta
                    start_date = (datetime.now() - timedelta(days=years_back * 365)).strftime('%Y-%m-%d')
                    
                    # Fetch historical data
                    backtester.fetch_historical_data(start_date)
                    
                    # Calculate portfolio history
                    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q"}
                    backtester.calculate_portfolio_history(freq_map[rebalance_freq])
                    
                    # Fetch benchmark data
                    backtester.fetch_benchmark_data()
                    
                    # Calculate performance metrics
                    metrics = backtester.calculate_performance_metrics()
                    
                    # Store in session state for other tabs
                    st.session_state.backtester = backtester
                    st.session_state.metrics = metrics
                    
                    st.success("✅ Analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
    
    with col2:
        # Display metrics if available
        if 'metrics' in st.session_state:
            st.markdown("**Performance Metrics:**")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Total Return", f"{st.session_state.metrics['Total_Return']:.2f}%")
                st.metric("Annualized Return", f"{st.session_state.metrics['Annualized_Return']:.2f}%")
            
            with col_b:
                st.metric("Sharpe Ratio", f"{st.session_state.metrics['Sharpe_Ratio']:.2f}")
                st.metric("Sortino Ratio", f"{st.session_state.metrics['Sortino_Ratio']:.2f}")
            
            with col_c:
                st.metric("Max Drawdown", f"{st.session_state.metrics['Max_Drawdown']:.2f}%")
                st.metric("Calmar Ratio", f"{st.session_state.metrics['Calmar_Ratio']:.2f}")
            
            with col_d:
                st.metric("Win Rate", f"{st.session_state.metrics['Win_Rate']:.1f}%")
                st.metric("Best/Worst Day", f"{st.session_state.metrics['Best_Day']:.2f}% / {st.session_state.metrics['Worst_Day']:.2f}%")
        
        # Display performance chart if available
        if 'backtester' in st.session_state:
            st.markdown("**Performance Chart:**")
            performance_chart = st.session_state.backtester.create_performance_chart()
            st.plotly_chart(performance_chart, use_container_width=True)

with tab2:
    st.markdown("### Monte Carlo Simulation")
    st.markdown("Simulate thousands of possible future scenarios for your portfolio.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Simulation Parameters:**")
        num_simulations = st.slider("Number of Simulations", 100, 5000, 1000)
        days_ahead = st.slider("Days to Simulate", 30, 730, 252)
        
        if st.button("🎲 Run Monte Carlo Simulation"):
            if 'backtester' in st.session_state:
                with st.spinner("Running Monte Carlo simulation..."):
                    try:
                        monte_carlo_results = st.session_state.backtester.run_monte_carlo_simulation(
                            num_simulations, days_ahead
                        )
                        st.session_state.monte_carlo = monte_carlo_results
                        st.success("✅ Monte Carlo simulation completed!")
                    except Exception as e:
                        st.error(f"❌ Error during simulation: {str(e)}")
            else:
                st.warning("⚠️ Please run Performance Analysis first!")
    
    with col2:
        if 'monte_carlo' in st.session_state:
            stats = st.session_state.monte_carlo['statistics']
            
            st.markdown("**Simulation Results:**")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("Current Value", f"${st.session_state.monte_carlo['current_value']:,.2f}")
                st.metric("Probability of Loss", f"{stats['Probability_of_Loss']:.1f}%")
            
            with col_b:
                st.metric("P25 (Conservative)", f"${stats['P25']:,.2f}")
                st.metric("P50 (Median)", f"${stats['P50']:,.2f}")
            
            with col_c:
                st.metric("P75 (Optimistic)", f"${stats['P75']:,.2f}")
                st.metric("P95 (Best Case)", f"${stats['P95']:,.2f}")
            
            # Display Monte Carlo chart
            monte_carlo_chart = st.session_state.backtester.create_monte_carlo_chart(
                st.session_state.monte_carlo
            )
            st.plotly_chart(monte_carlo_chart, use_container_width=True)

with tab3:
    st.markdown("### Strategy Comparison")
    st.markdown("Compare your portfolio against different strategies and benchmarks.")
    
    if 'backtester' in st.session_state and 'metrics' in st.session_state:
        # Create comparison table
        comparison_data = {
            'Strategy': ['Your Portfolio', 'S&P 500 (Benchmark)', '60/40 Stock/Bond', 'Tech-Heavy Portfolio'],
            'Annual Return': [
                f"{st.session_state.metrics['Annualized_Return']:.2f}%",
                "~10%",  # Historical S&P 500
                "~8%",   # Typical 60/40
                "~12%"   # Typical tech-heavy
            ],
            'Volatility': [
                f"{st.session_state.metrics['Annualized_Volatility']:.2f}%",
                "~15%",
                "~10%",
                "~25%"
            ],
            'Sharpe Ratio': [
                f"{st.session_state.metrics['Sharpe_Ratio']:.2f}",
                "~0.67",
                "~0.80",
                "~0.48"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Risk-Return scatter plot
        fig_risk_return = px.scatter(
            x=[st.session_state.metrics['Annualized_Volatility'], 15, 10, 25],
            y=[st.session_state.metrics['Annualized_Return'], 10, 8, 12],
            text=['Your Portfolio', 'S&P 500', '60/40', 'Tech-Heavy'],
            title="Risk vs Return Comparison"
        )
        
        fig_risk_return.update_traces(
            textposition="top center",
            marker=dict(size=15, color=['blue', 'red', 'green', 'orange'])
        )
        
        fig_risk_return.update_layout(
            xaxis_title="Volatility (%)",
            yaxis_title="Annual Return (%)",
            height=400
        )
        
        st.plotly_chart(fig_risk_return, use_container_width=True)
        
    else:
        st.info("💡 Run Performance Analysis first to see strategy comparison!")

# --- END BACKTESTING SECTION ---