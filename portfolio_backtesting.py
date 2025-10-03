import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PortfolioBacktester:
    """
    Advanced Portfolio Backtesting Engine
    Supports multiple strategies, risk metrics, and comprehensive analysis
    """
    
    def __init__(self, portfolio_df: pd.DataFrame):
        self.portfolio_df = portfolio_df.copy()
        self.historical_data = {}
        self.portfolio_history = pd.DataFrame()
        self.benchmark_data = {}
        
    def fetch_historical_data(self, start_date: str, end_date: str = None) -> Dict:
        """
        Fetch historical data for all portfolio holdings
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: today)
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"📊 Fetching historical data from {start_date} to {end_date}...")
        
        for ticker in self.portfolio_df['Ticker']:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    self.historical_data[ticker] = data
                    print(f"✅ {ticker}: {len(data)} days of data")
                else:
                    print(f"❌ {ticker}: No data available")
            except Exception as e:
                print(f"❌ {ticker}: Error - {str(e)}")
                
        return self.historical_data
    
    def calculate_portfolio_history(self, rebalance_frequency: str = 'M') -> pd.DataFrame:
        """
        Calculate portfolio value over time with optional rebalancing
        
        Args:
            rebalance_frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), 'Q' (quarterly)
        """
        if not self.historical_data:
            raise ValueError("No historical data available. Run fetch_historical_data() first.")
            
        print(f"🔄 Calculating portfolio history with {rebalance_frequency} rebalancing...")
        
        # Get all dates from the first ticker
        first_ticker = list(self.historical_data.keys())[0]
        all_dates = self.historical_data[first_ticker].index
        
        portfolio_values = []
        portfolio_weights = self._calculate_initial_weights()
        
        for date in all_dates:
            daily_value = 0
            valid_holdings = 0
            
            for ticker, weight in portfolio_weights.items():
                if ticker in self.historical_data:
                    if date in self.historical_data[ticker].index:
                        price = self.historical_data[ticker].loc[date, 'Close']
                        shares = self.portfolio_df[self.portfolio_df['Ticker'] == ticker]['Shares'].iloc[0]
                        
                        # ← ADD THESE LINES:
                        # Check for NaN or invalid values
                        # Convert to scalar values first to avoid Series comparison issues
                        try:
                            # Handle both Series and scalar values
                            if hasattr(price, 'iloc'):  # If it's a Series
                                price_val = float(price.iloc[0])
                            else:  # If it's already a scalar
                                price_val = float(price)
                                
                            if hasattr(shares, 'iloc'):  # If it's a Series
                                shares_val = float(shares.iloc[0])
                            else:  # If it's already a scalar
                                shares_val = float(shares)
                                
                        except (ValueError, TypeError, IndexError):
                            continue

                        # Now check with scalar values
                        if pd.isna(price_val) or pd.isna(shares_val) or price_val <= 0 or shares_val <= 0:
                            continue
                            
                        # Convert to float explicitly
                        price_float = price_val
                        shares_float = shares_val
                        holding_value = price_float * shares_float
                        
                        daily_value += holding_value  # ← CHANGED: Now using holding_value instead of price * shares
                        valid_holdings += 1
            
            if valid_holdings > 0:
                portfolio_values.append({
                    'Date': date,
                    'Portfolio_Value': daily_value,
                    'Valid_Holdings': valid_holdings
                })
        
        self.portfolio_history = pd.DataFrame(portfolio_values)
        self.portfolio_history.set_index('Date', inplace=True)
        
        print(f"✅ Portfolio history calculated: {len(self.portfolio_history)} data points")
        return self.portfolio_history
    
    def _calculate_initial_weights(self) -> Dict:
        """Calculate initial portfolio weights based on current values"""
        total_value = (self.portfolio_df['Shares'] * self.portfolio_df['Current Price']).sum()
        weights = {}
        
        for _, row in self.portfolio_df.iterrows():
            ticker = row['Ticker']
            value = row['Shares'] * row['Current Price']
            weights[ticker] = value / total_value
            
        return weights
    
    def fetch_benchmark_data(self, benchmarks: List[str] = None, start_date: str = None) -> Dict:
        """
        Fetch benchmark data for comparison
        
        Args:
            benchmarks: List of benchmark tickers (default: ['^GSPC', '^IXIC'])
            start_date: Start date for benchmark data
        """
        if benchmarks is None:
            benchmarks = ['^GSPC', '^IXIC']  # S&P 500 and Nasdaq
            
        if start_date is None and hasattr(self, 'portfolio_history') and not self.portfolio_history.empty:
            start_date = self.portfolio_history.index[0].strftime('%Y-%m-%d')
        elif start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
        print(f"📈 Fetching benchmark data from {start_date}...")
        
        benchmark_names = {'^GSPC': 'S&P 500', '^IXIC': 'Nasdaq', '^DJI': 'Dow Jones'}
        
        for benchmark in benchmarks:
            try:
                data = yf.download(benchmark, start=start_date, progress=False)
                if not data.empty:
                    name = benchmark_names.get(benchmark, benchmark)
                    self.benchmark_data[name] = data
                    print(f"✅ {name}: {len(data)} days of data")
                else:
                    print(f"❌ {benchmark}: No data available")
            except Exception as e:
                print(f"❌ {benchmark}: Error - {str(e)}")
                
        return self.benchmark_data
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if self.portfolio_history.empty:
            raise ValueError("No portfolio history available. Run calculate_portfolio_history() first.")
            
        print("📊 Calculating performance metrics...")
        
        # Basic returns
        returns = self.portfolio_history['Portfolio_Value'].pct_change().dropna()
        
        # Annualized metrics
        trading_days = 252
        years = len(returns) / trading_days
        
        metrics = {
            'Total_Return': (self.portfolio_history['Portfolio_Value'].iloc[-1] / self.portfolio_history['Portfolio_Value'].iloc[0] - 1) * 100,
            'Annualized_Return': ((1 + returns.mean()) ** trading_days - 1) * 100,
            'Annualized_Volatility': returns.std() * np.sqrt(trading_days) * 100,
            'Sharpe_Ratio': (returns.mean() * trading_days) / (returns.std() * np.sqrt(trading_days)),
            'Max_Drawdown': self._calculate_max_drawdown(),
            'Calmar_Ratio': 0,  # Will be calculated after max drawdown
            'Sortino_Ratio': self._calculate_sortino_ratio(returns),
            'Win_Rate': (returns > 0).mean() * 100,
            'Best_Day': returns.max() * 100,
            'Worst_Day': returns.min() * 100
        }
        
        # Calculate Calmar Ratio
        if metrics['Max_Drawdown'] != 0:
            metrics['Calmar_Ratio'] = metrics['Annualized_Return'] / abs(metrics['Max_Drawdown'])
            
        print("✅ Performance metrics calculated")
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        peak = self.portfolio_history['Portfolio_Value'].expanding().max()
        drawdown = (self.portfolio_history['Portfolio_Value'] - peak) / peak
        return drawdown.min() * 100
    
    def _calculate_sortino_ratio(self, returns: pd.Series, target_return: float = 0) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return np.inf
        
        downside_deviation = downside_returns.std() * np.sqrt(252)
        if downside_deviation == 0:
            return np.inf
            
        return (returns.mean() * 252 - target_return) / downside_deviation
    
    def create_performance_chart(self, include_benchmarks: bool = True) -> go.Figure:
        """
        Create comprehensive performance chart with portfolio and benchmarks
        """
        if self.portfolio_history.empty:
            raise ValueError("No portfolio history available.")
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Performance', 'Drawdown', 'Monthly Returns', 'Rolling Sharpe Ratio'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Portfolio performance
        fig.add_trace(
            go.Scatter(
                x=self.portfolio_history.index,
                y=self.portfolio_history['Portfolio_Value'],
                mode='lines',
                name='Your Portfolio',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add benchmarks if available
        if include_benchmarks and self.benchmark_data:
            for benchmark_name, benchmark_data in self.benchmark_data.items():
                # Normalize benchmark to start at same value as portfolio
                benchmark_normalized = benchmark_data['Close'] / benchmark_data['Close'].iloc[0] * self.portfolio_history['Portfolio_Value'].iloc[0]
                
                fig.add_trace(
                    go.Scatter(
                        x=benchmark_data.index,
                        y=benchmark_normalized,
                        mode='lines',
                        name=benchmark_name,
                        line=dict(width=1, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # Drawdown chart
        peak = self.portfolio_history['Portfolio_Value'].expanding().max()
        drawdown = (self.portfolio_history['Portfolio_Value'] - peak) / peak * 100
        
        fig.add_trace(
            go.Scatter(
                x=self.portfolio_history.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(color='red')
            ),
            row=1, col=2
        )
        
        # Monthly returns heatmap
        returns = self.portfolio_history['Portfolio_Value'].pct_change().dropna()
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
        
        fig.add_trace(
            go.Scatter(
                x=monthly_returns.index,
                y=monthly_returns.values,
                mode='markers',
                name='Monthly Returns',
                marker=dict(
                    size=8,
                    color=monthly_returns.values,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(
                        title="Return %",
                        x=1.15,
                        len=0.8
                    )
                )
            ),
            row=2, col=1
        )
        
        # Rolling Sharpe Ratio
        rolling_sharpe = returns.rolling(window=30).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() != 0 else 0
        )
        
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='30-Day Rolling Sharpe',
                line=dict(color='green')
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Portfolio Performance Analysis',
            height=800,
            showlegend=True,
            template='plotly_white',
            legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left", 
            x=1.02
        ),
        margin=dict(l=60, r=120, t=80, b=60)
        )
        
        return fig
    
    def run_monte_carlo_simulation(self, num_simulations: int = 1000, days_ahead: int = 252) -> Dict:
        """
        Run Monte Carlo simulation for portfolio forecasting
        
        Args:
            num_simulations: Number of simulation paths
            days_ahead: Days to simulate into the future
        """
        if self.portfolio_history.empty:
            raise ValueError("No portfolio history available.")
            
        print(f"🎲 Running {num_simulations} Monte Carlo simulations...")
        
        returns = self.portfolio_history['Portfolio_Value'].pct_change().dropna()
        current_value = self.portfolio_history['Portfolio_Value'].iloc[-1]
        
        # Calculate parameters
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Generate future dates
        last_date = self.portfolio_history.index[-1]
        future_dates = pd.date_range(start=last_date, periods=days_ahead + 1, freq='D')[1:]
        
        # Run simulations
        simulation_paths = []
        for i in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, volatility, days_ahead)
            # Calculate cumulative path
            path = [current_value]
            for ret in random_returns:
                path.append(path[-1] * (1 + ret))
            simulation_paths.append(path[1:])  # Remove initial value
        
        simulation_df = pd.DataFrame(simulation_paths, columns=future_dates).T
        
        # Calculate statistics
        final_values = simulation_df.iloc[-1]
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        stats = {f'P{p}': np.percentile(final_values, p) for p in percentiles}
        
        # Calculate probability of loss
        stats['Probability_of_Loss'] = (final_values < current_value).mean() * 100
        
        print("✅ Monte Carlo simulation completed")
        
        return {
            'simulation_data': simulation_df,
            'statistics': stats,
            'current_value': current_value,
            'future_dates': future_dates
        }
    
    def create_monte_carlo_chart(self, monte_carlo_results: Dict, show_percentiles: bool = True) -> go.Figure:
        """Create Monte Carlo simulation visualization"""
        simulation_df = monte_carlo_results['simulation_data']
        stats = monte_carlo_results['statistics']
        
        fig = go.Figure()
        
        # Add simulation paths (sample)
        sample_paths = simulation_df.iloc[:, :100]  # Show first 100 paths
        for col in sample_paths.columns:
            fig.add_trace(
                go.Scatter(
                    x=simulation_df.index,
                    y=simulation_df[col],
                    mode='lines',
                    opacity=0.1,
                    line=dict(color='lightblue', width=1),
                    showlegend=False
                )
            )
        
        # Add percentile lines
        if show_percentiles:
            for percentile, value in stats.items():
                if percentile.startswith('P'):
                    fig.add_trace(
                        go.Scatter(
                            x=simulation_df.index,
                            y=[value] * len(simulation_df),
                            mode='lines',
                            name=f'{percentile}',
                            line=dict(width=2, dash='dash')
                        )
                    )
        
        # Add current value line
        fig.add_trace(
            go.Scatter(
                x=simulation_df.index,
                y=[monte_carlo_results['current_value']] * len(simulation_df),
                mode='lines',
                name='Current Value',
                line=dict(color='red', width=3)
            )
        )
        
        fig.update_layout(
            title=f'Monte Carlo Simulation Results<br>Probability of Loss: {stats["Probability_of_Loss"]:.1f}%',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=500,
            template='plotly_white'
        )
        
        return fig

# Example usage function
def demonstrate_backtesting(portfolio_df: pd.DataFrame):
    """
    Demonstrate the backtesting functionality
    """
    print("🚀 Starting Portfolio Backtesting Demo...")
    
    # Initialize backtester
    backtester = PortfolioBacktester(portfolio_df)
    
    # Fetch historical data (last 2 years)
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    backtester.fetch_historical_data(start_date)
    
    # Calculate portfolio history
    backtester.calculate_portfolio_history()
    
    # Fetch benchmark data
    backtester.fetch_benchmark_data()
    
    # Calculate performance metrics
    metrics = backtester.calculate_performance_metrics()
    
    print("\n📊 Performance Metrics:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")
    
    return backtester, metrics