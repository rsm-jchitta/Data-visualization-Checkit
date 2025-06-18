import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fetch_financial_data(ticker_symbol):
    """Fetch and process financial data for a given ticker"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Fetch quarterly financials (raw data)
        income_quarterly = ticker.quarterly_income_stmt
        cash_flow_quarterly = ticker.quarterly_cashflow
        balance_sheet_quarterly = ticker.quarterly_balance_sheet
        
        # Check if data is available
        if income_quarterly.empty:
            return None, "Income statement data not available"
        
        # Gross Margin Calculation
        if 'Gross Profit' in income_quarterly.index:
            income_quarterly.loc['Gross Margin %'] = (
                income_quarterly.loc['Gross Profit'] * 100.0 / income_quarterly.loc['Total Revenue']
            )
        elif 'Total Revenue' in income_quarterly.index and 'Cost Of Revenue' in income_quarterly.index:
            income_quarterly.loc['Gross Margin %'] = (
                (income_quarterly.loc['Total Revenue'] - income_quarterly.loc['Cost Of Revenue']) * 100.0 / income_quarterly.loc['Total Revenue']
            )
        else:
            # Add placeholder row if calculation not possible
            income_quarterly.loc['Gross Margin %'] = np.nan
        
        # Operating Expense calculation
        if 'Operating Expense' not in income_quarterly.index:
            if ('Selling General And Administration' in income_quarterly.index) and ('Research And Development' in income_quarterly.index):
                income_quarterly.loc['Operating Expense'] = (
                    income_quarterly.loc['Selling General And Administration'] + income_quarterly.loc['Research And Development']
                )
            else:
                income_quarterly.loc['Operating Expense'] = np.nan
        
        # EBIT calculation
        if 'EBIT' not in income_quarterly.index:
            if 'Operating Income' in income_quarterly.index:
                income_quarterly.loc['EBIT'] = income_quarterly.loc['Operating Income']
            else:
                income_quarterly.loc['EBIT'] = np.nan
        
        # Free Cash Flow calculation
        if 'Free Cash Flow' not in cash_flow_quarterly.index:
            if ('Operating Cash Flow' in cash_flow_quarterly.index) and ('Capital Expenditure' in cash_flow_quarterly.index):
                cash_flow_quarterly.loc['Free Cash Flow'] = (
                    cash_flow_quarterly.loc['Operating Cash Flow'] + cash_flow_quarterly.loc['Capital Expenditure']
                )
            else:
                cash_flow_quarterly.loc['Free Cash Flow'] = np.nan
        
        # Create filtered datasets with error handling
        metrics_to_include = ['Total Revenue', 'Operating Expense', 'Gross Margin %', 'EBIT', 'Net Income']
        filtered_income_data = []
        
        for metric in metrics_to_include:
            if metric in income_quarterly.index:
                filtered_income_data.append(income_quarterly.loc[metric].iloc[:5])
            else:
                # Create a series with NaN values if metric doesn't exist
                filtered_income_data.append(pd.Series([np.nan] * min(5, len(income_quarterly.columns)), 
                                                    index=income_quarterly.columns[:min(5, len(income_quarterly.columns))], 
                                                    name=metric))
        
        filtered_income = pd.DataFrame(filtered_income_data)
        
        # Handle Free Cash Flow
        if 'Free Cash Flow' in cash_flow_quarterly.index:
            filtered_cash_flow = cash_flow_quarterly.loc[['Free Cash Flow']].iloc[:, :5]
        else:
            # Create placeholder if Free Cash Flow doesn't exist
            filtered_cash_flow = pd.DataFrame([pd.Series([np.nan] * min(5, len(income_quarterly.columns)), 
                                                       index=income_quarterly.columns[:min(5, len(income_quarterly.columns))], 
                                                       name='Free Cash Flow')])
        
        # Combine the data
        result = pd.concat([filtered_income, filtered_cash_flow], axis=0)
        
        # Convert columns to period format
        result.columns = result.columns.to_period('Q')
        
        return result, None
        
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

def create_visualization(result_df, ticker_symbol):
    """Create the revenue and gross margin visualization"""
    try:
        # Check if required data is available
        if 'Total Revenue' not in result_df.index or 'Gross Margin %' not in result_df.index:
            st.warning("Required data for visualization not available")
            return
        
        # Transpose the DataFrame
        df_transposed = result_df.T
        # Convert to string format (if it's already a PeriodIndex, just convert to string)
        df_transposed.index = df_transposed.index.astype(str)
        
        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bar chart for Total Revenue
        revenue_data = df_transposed['Total Revenue'].dropna()
        if not revenue_data.empty:
            ax1.bar(revenue_data.index, revenue_data.values, color='skyblue', alpha=0.7, label='Total Revenue')
            ax1.set_ylabel('Total Revenue (in millions)', fontsize=12, color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Format y-axis labels
            y_ticks = ax1.get_yticks()
            ax1.set_yticklabels([f'{int(tick/1e6)}M' if tick != 0 else '0' for tick in y_ticks])
        
        ax1.set_xlabel('Quarter', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Line chart for Gross Margin %
        ax2 = ax1.twinx()
        margin_data = df_transposed['Gross Margin %'].dropna()
        if not margin_data.empty:
            ax2.plot(margin_data.index, margin_data.values, color='red', marker='o', linewidth=2, markersize=6, label='Gross Margin %')
            ax2.set_ylabel('Gross Margin %', fontsize=12, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        
        # Title and legend
        plt.title(f'{ticker_symbol} - Total Revenue and Gross Margin % by Quarter', fontsize=14, fontweight='bold')
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        fig.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.1, 0.9))
        
        fig.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")

def format_financial_data(df):
    """Format financial data for display - all numbers in millions except gross margin"""
    formatted_df = df.copy()
    
    # Format data for display
    for col in formatted_df.columns:
        for idx in formatted_df.index:
            value = formatted_df.loc[idx, col]
            if pd.isna(value):
                formatted_df.loc[idx, col] = "Data not available"
            elif idx == 'Gross Margin %':
                # Keep gross margin as percentage
                formatted_df.loc[idx, col] = f"{value:.2f}%" if not pd.isna(value) else "Data not available"
            elif isinstance(value, (int, float)):
                # Convert all other financial numbers to millions, round to integer, and add commas
                formatted_df.loc[idx, col] = f"{round(value/1e6):,}"
    
    return formatted_df

# Streamlit App
def main():
    st.set_page_config(page_title="Financial Dashboard", page_icon="📊", layout="wide")
    
    st.title("📊 Financial Dashboard")
    st.markdown("Enter a stock ticker to view key financial metrics and visualizations")
    st.markdown("**Note**: All financial figures are displayed in millions of dollars, except Gross Margin which is shown as a percentage.")
    
    # Sidebar for user input
    st.sidebar.header("Settings")
    ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., TSLA, AAPL, MSFT)", value="TSLA").upper()
    
    if st.sidebar.button("Analyze", type="primary") or ticker_input:
        if ticker_input:
            with st.spinner(f"Fetching data for {ticker_input}..."):
                result_df, error = fetch_financial_data(ticker_input)
            
            if error:
                st.error(f"❌ {error}")
                st.info("Please try a different ticker symbol or check if the company has publicly available financial data.")
            else:
                st.success(f"✅ Data successfully loaded for {ticker_input}")
                
                # Display company info
                try:
                    ticker_obj = yf.Ticker(ticker_input)
                    info = ticker_obj.info
                    company_name = info.get('longName', ticker_input)
                    st.subheader(f"{company_name} ({ticker_input})")
                except:
                    st.subheader(f"{ticker_input}")
                
                # Create two columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📋 Key Financial Metrics (in millions $)")
                    formatted_df = format_financial_data(result_df)
                    st.dataframe(formatted_df, use_container_width=True)
                    
                    # Download button for the data
                    csv = result_df.to_csv()
                    st.download_button(
                        label="📥 Download Data as CSV",
                        data=csv,
                        file_name=f"{ticker_input}_financial_data.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.subheader("📈 Revenue & Margin Visualization")
                    create_visualization(result_df, ticker_input)
                
                # Additional metrics summary
                st.subheader("📊 Quick Summary")
                col_a, col_b, col_c, col_d = st.columns(4)
                
                try:
                    latest_revenue = result_df.loc['Total Revenue'].iloc[0]
                    latest_margin = result_df.loc['Gross Margin %'].iloc[0]
                    latest_net_income = result_df.loc['Net Income'].iloc[0]
                    latest_fcf = result_df.loc['Free Cash Flow'].iloc[0]
                    
                    with col_a:
                        st.metric("Latest Revenue (M$)", 
                                f"{round(latest_revenue/1e6):,}" if not pd.isna(latest_revenue) else "N/A")
                    with col_b:
                        st.metric("Latest Gross Margin", 
                                f"{latest_margin:.1f}%" if not pd.isna(latest_margin) else "N/A")
                    with col_c:
                        st.metric("Latest Net Income (M$)", 
                                f"{round(latest_net_income/1e6):,}" if not pd.isna(latest_net_income) else "N/A")
                    with col_d:
                        st.metric("Latest Free Cash Flow (M$)", 
                                f"{round(latest_fcf/1e6):,}" if not pd.isna(latest_fcf) else "N/A")
                except:
                    st.info("Summary metrics unavailable")
        else:
            st.info("👆 Please enter a stock ticker in the sidebar to get started")
    
    # Footer
    st.markdown("---")
    st.markdown("💡 **Tip**: Try popular tickers like TSLA, AAPL, MSFT, GOOGL, AMZN, META, NVDA")

if __name__ == "__main__":
    main()
