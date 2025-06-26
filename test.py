
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import io
import base64
import hashlib
from datetime import datetime

# # File names for saving CSVs
# income_csv_file = 'income_data.csv'
# sector_csv_file = 'sector_data.csv'



# --- 1. Function to generate monthly expense CSV report for sharing ---
def generate_monthly_expense_report_csv(year, month):
    _, sector_data = load_data()
    if sector_data is None:
        return None

    monthly_data = sector_data[
        (sector_data['Year'] == year) & (sector_data['Month'] == month)
    ]
    if monthly_data.empty:
        return None

    csv_buffer = io.StringIO()
    monthly_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer

# --- 2. Savings Goal Data & Achievement checks ---
# Use session state to persist goal
if 'savings_goal' not in st.session_state:
    st.session_state.savings_goal = 0.0

def check_achievements():
    savings_data = calculate_savings()
    badges = []

    if savings_data is None or savings_data.empty:
        return badges

    # Badge 1: Saved positive for 3 consecutive months
    positive_savings = savings_data['Savings'] > 0
    # Check for any 3 consecutive True in positive_savings
    consec = 0
    for val in positive_savings:
        if val:
            consec += 1
            if consec >= 3:
                badges.append("🏅 Consistent Saver: Positive savings for 3+ months")
                break
        else:
            consec = 0

    # Badge 2: Stayed under budget (if user sets goal as a max spending, for example)
    # Here we’ll assume savings_goal is minimum saving target; if savings >= goal in last month
    if st.session_state.savings_goal > 0:
        last_savings = savings_data['Savings'].iloc[-1]
        if last_savings >= st.session_state.savings_goal:
            badges.append("🎯 Goal Achiever: Savings goal met or exceeded this month")

    return badges



# --- User login system with common user CSV ---
users_file = 'users.csv'

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(users_file):
        return pd.read_csv(users_file)
    else:
        return pd.DataFrame(columns=['username', 'password_hash'])

def save_user(username, password):
    users = load_users()
    if username in users['username'].values:
        return False  # user exists
    pw_hash = hash_password(password)
    new_user = pd.DataFrame({'username':[username], 'password_hash':[pw_hash]})
    new_user.to_csv(users_file, mode='a', header=not os.path.exists(users_file), index=False)
    return True

def check_credentials(username, password):
    users = load_users()
    pw_hash = hash_password(password)
    user_row = users[(users['username'] == username) & (users['password_hash'] == pw_hash)]
    return not user_row.empty

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

if not st.session_state.logged_in:
    st.title("Login to Personal Financial Tracker")
    choice = st.selectbox("Select Option", ["Login", "Register"])

    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if check_credentials(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")

    elif choice == "Register":
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type='password')
        confirm_password = st.text_input("Confirm Password", type='password')
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif save_user(new_username, new_password):
                st.success("User registered! Please login now.")
            else:
                st.error("Username already exists")

    st.stop()

# --- Use shared CSV files ---
income_csv_file = 'income.csv'
sector_csv_file = 'sector.csv'

# --- Save income data with username ---
def save_income_to_csv(income, date, deposit, username):
    df = pd.DataFrame({
        'username': [username],
        'Income Amount': [income],
        'Date': [date],
        'Deposit Amount': [deposit]
    })
    if os.path.exists(income_csv_file):
        df.to_csv(income_csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(income_csv_file, mode='w', header=True, index=False)

# --- Save sector data with username ---
def save_sector_to_csv(sector, amount, date, username):
    df = pd.DataFrame({
        'username': [username],
        'Sector': [sector],
        'Amount': [amount],
        'Date': [date]
    })
    if os.path.exists(sector_csv_file):
        df.to_csv(sector_csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(sector_csv_file, mode='w', header=True, index=False)

# --- Load user-specific income and sector data ---
def load_data():
    income_data, sector_data = None, None

    if os.path.exists(income_csv_file):
        try:
            income_data = pd.read_csv(income_csv_file)
            income_data.columns = income_data.columns.str.strip()
            income_data = income_data[income_data['username'] == st.session_state.username]
            if 'Date' in income_data.columns:
                income_data['Date'] = pd.to_datetime(income_data['Date'], errors='coerce')
                income_data['Year'] = income_data['Date'].dt.year
                income_data['Month'] = income_data['Date'].dt.strftime('%B')
            else:
                st.error("'Date' column not found in income.csv")
        except Exception as e:
            st.error(f"Error reading income CSV: {e}")

    if os.path.exists(sector_csv_file):
        try:
            sector_data = pd.read_csv(sector_csv_file)
            sector_data.columns = sector_data.columns.str.strip()
            sector_data = sector_data[sector_data['username'] == st.session_state.username]
            if 'Date' in sector_data.columns:
                sector_data['Date'] = pd.to_datetime(sector_data['Date'], errors='coerce')
                sector_data['Year'] = sector_data['Date'].dt.year
                sector_data['Month'] = sector_data['Date'].dt.strftime('%B')
            else:
                st.error("'Date' column not found in sector.csv")
        except Exception as e:
            st.error(f"Error reading sector CSV: {e}")

    return income_data, sector_data




# Function to calculate monthly savings
def calculate_savings():
    income_data, sector_data = load_data()
    
    if income_data is None or sector_data is None:
        return None
    
    # Group income and expenses by month
    monthly_income = income_data.groupby('Month')['Income Amount'].sum()
    monthly_expenses = sector_data.groupby('Month')['Amount'].sum()

    # Align income and expenses by month
    combined_data = pd.DataFrame({'Income': monthly_income, 'Expenses': monthly_expenses}).fillna(0)
    
    # Calculate savings
    combined_data['Savings'] = combined_data['Income'] - combined_data['Expenses']
    
    return combined_data

# Function to calculate total and current balance
def calculate_balance():
    savings_data = calculate_savings()

    if savings_data is None or savings_data.empty:
        return None, None

    # Calculate total income, total expenses, and total balance
    total_income = savings_data['Income'].sum()
    total_expenses = savings_data['Expenses'].sum()
    total_balance = total_income - total_expenses

    # Calculate current balance as the most recent month's savings
    current_balance = savings_data['Savings'].iloc[-1] if not savings_data.empty else 0

    return total_balance, current_balance

# Function to plot savings graph (amount vs month)
def plot_savings_graph():
    savings_data = calculate_savings()
    
    if savings_data is not None and not savings_data.empty:
        # Sort savings by month order
        months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                        'July', 'August', 'September', 'October', 'November', 'December']
        savings_data = savings_data.reindex(months_order)
        
        # Create the line chart for savings
        fig = px.line(x=savings_data.index, y=savings_data['Savings'], 
                      title='Savings by Month', 
                      labels={'x': 'Month', 'y': 'Savings (TK)'},
                      markers=True, 
                      color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig)
    else:
        st.warning("No sufficient data available to calculate savings.")

# Function to plot Expense vs Income
def plot_expense_vs_income(selected_year):
    income_data, sector_data = load_data()

    if income_data is None or sector_data is None:
        st.warning("No income or expense data available.")
        return

    # Filter data by year
    income_year_data = income_data[income_data['Year'] == selected_year]
    sector_year_data = sector_data[sector_data['Year'] == selected_year]

    if income_year_data.empty or sector_year_data.empty:
        st.warning(f"No data available for {selected_year}.")
        return

    # Group income and expenses by month
    monthly_income = income_year_data.groupby('Month')['Income Amount'].sum()
    monthly_expenses = sector_year_data.groupby('Month')['Amount'].sum()

    # Merge income and expenses into a single DataFrame for plotting
    combined_data = pd.DataFrame({'Income': monthly_income, 'Expenses': monthly_expenses}).reset_index()

    # Plot both Income and Expenses
    fig = px.bar(combined_data, x='Month', y=['Income', 'Expenses'],
                 title=f'Income vs Expenses for {selected_year}',
                 labels={'value': 'Amount (Tk)', 'Month': 'Month'},
                 barmode='group',
                 color_discrete_map={'Income': 'green', 'Expenses': 'red'})
    st.plotly_chart(fig)

# Function to plot yearly expense data
def plot_yearly_data(selected_year):
    _, sector_data = load_data()
    
    if sector_data is None or selected_year not in sector_data['Year'].values:
        st.warning(f"No data available for {selected_year}.")
        return
    
    # Filter sector data by the selected year
    filtered_data = sector_data[sector_data['Year'] == selected_year]
    
    # Group by Sector and sum amounts for the selected year
    sector_summary = filtered_data.groupby('Sector')['Amount'].sum().reset_index()

    # Plot the bar chart for yearly data
    fig_bar = px.bar(sector_summary, x='Sector', y='Amount', 
                     title=f'Expenses by Sector in {selected_year}',
                     color='Amount', 
                     text='Amount', 
                     color_continuous_scale=px.colors.sequential.Viridis)
    st.plotly_chart(fig_bar)
    
    # Plot the pie chart for yearly data
    fig_pie = px.pie(sector_summary, names='Sector', values='Amount', 
                     title=f'Expense Distribution in {selected_year}', 
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_pie)

# Function to plot monthly data for a specific year and month
def plot_monthly_data(selected_year, selected_month):
    _, sector_data = load_data()
    
    if sector_data is None or selected_year not in sector_data['Year'].values:
        st.warning(f"No data available for {selected_year}.")
        return
    
    # Filter sector data by the selected year and month
    filtered_data = sector_data[(sector_data['Year'] == selected_year) & (sector_data['Month'] == selected_month)]
    
    if filtered_data.empty:
        st.warning(f"No data available for {selected_month}, {selected_year}.")
        return
    
    # Group by Sector and sum amounts for the selected month and year
    sector_summary = filtered_data.groupby('Sector')['Amount'].sum().reset_index()

    # Plot the bar chart for the selected month
    fig_bar = px.bar(sector_summary, x='Sector', y='Amount', 
                     title=f'Expenses by Sector in {selected_month}, {selected_year}',
                     color='Amount', 
                     text='Amount', 
                     color_continuous_scale=px.colors.sequential.Rainbow)
    st.plotly_chart(fig_bar)
    
    # Plot the pie chart for the selected month
    fig_pie = px.pie(sector_summary, names='Sector', values='Amount', 
                     title=f'Expense Distribution in {selected_month}, {selected_year}', 
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig_pie)

# Function to predict future expenses
def predict_expenses(months_ahead=3):
    if os.path.exists(sector_csv_file):
        df_sector = pd.read_csv(sector_csv_file)
        df_sector['Date'] = pd.to_datetime(df_sector['Date'])

        # Prepare data for linear regression
        df_sector['Month'] = df_sector['Date'].dt.month
        df_sector['Year'] = df_sector['Date'].dt.year
        df_grouped = df_sector.groupby(['Year', 'Month'])['Amount'].sum().reset_index()

        # Create numerical features for regression
        df_grouped['Date_Ordinal'] = pd.to_datetime(df_grouped[['Year', 'Month']].assign(DAY=1)).map(datetime.toordinal)

        # Fit linear regression model
        X = df_grouped['Date_Ordinal'].values.reshape(-1, 1)
        y = df_grouped['Amount'].values

        model = LinearRegression()
        model.fit(X, y)

        # Predict future expenses
        future_dates = [X[-1][0] + i * 30 for i in range(1, months_ahead + 1)]  # Predict next months
        predictions = model.predict(np.array(future_dates).reshape(-1, 1))

        return predictions
    else:
        st.warning("Sector data file does not exist.")
        return None

# Streamlit App
st.title("Personal Financial Tracker")

# Sidebar navigation
page = st.sidebar.radio("Select a Page", ["Income and Deposit", "Expense", "Visualize Data", "Savings", "Expense Prediction", "Balance"])

# Income and Deposit Tracker Page
if page == "Income and Deposit":
    st.header("Income and Deposit Tracker")

    # Sidebar inputs for income data
    income = st.sidebar.number_input("Income Amount", min_value=0.0, format="%.2f")
    income_date = st.sidebar.date_input("Date", value=datetime.today())
    deposit = st.sidebar.number_input("Deposit Amount", min_value=0.0, format="%.2f")

    # Save income data
    if st.sidebar.button("Save Income Data"):
        save_income_to_csv(income, income_date, deposit)
        st.sidebar.success("Income data saved!")

    # Display saved income data
    if os.path.exists(income_csv_file):
        st.subheader("Saved Income Data")
        df_income = pd.read_csv(income_csv_file)
        st.dataframe(df_income)
    else:
        st.info("No income data saved yet.")

# Expense Tracker Page
elif page == "Expense":
    st.header("Expense Tracker")

    # Fixed category list for expenses
    expense_categories = ['Rent', 'Entertainment', 'Utilities', 'Transportation', 'Groceries', 'Healthcare', 'Education', 'Miscellaneous']
    
    # Sidebar inputs for sector data
    sector = st.sidebar.selectbox("Select Expense Category", expense_categories)  # Dropdown for categories
    sector_amount = st.sidebar.number_input("Amount", min_value=0.0, format="%.2f")
    sector_date = st.sidebar.date_input("Date", value=datetime.today())

    # Save sector data
    if st.sidebar.button("Save Sector Data"):
        save_sector_to_csv(sector, sector_amount, sector_date)
        st.sidebar.success("Sector data saved!")

    # Display saved sector data
    if os.path.exists(sector_csv_file):
        st.subheader("Saved Sector Data")
        df_sector = pd.read_csv(sector_csv_file)
        st.dataframe(df_sector)
    else:
        st.info("No sector data saved yet.")

# Visualize Data Page (Yearly and Monthly data selection)
# Inside Visualize Data page, add option to share monthly report CSV

elif page == "Visualize Data":
    st.header("Data Visualization")

    view_type = st.radio("View By", ["Yearly", "Monthly", "Expense vs Income"])

    _, sector_data = load_data()
    available_years = sorted(sector_data['Year'].unique()) if sector_data is not None else []
    selected_year = st.selectbox("Select Year", available_years)

    if view_type == "Yearly":
        plot_yearly_data(selected_year)
    
    elif view_type == "Monthly":
        months = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
        selected_month = st.selectbox("Select Month", months)

        plot_monthly_data(selected_year, selected_month)

        # Add report sharing button
        csv_buffer = generate_monthly_expense_report_csv(selected_year, selected_month)
        if csv_buffer is not None:
            st.download_button(
                label=f"Download {selected_month} {selected_year} Expense Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f'expense_report_{selected_year}_{selected_month}.csv',
                mime='text/csv'
            )
        else:
            st.info(f"No data to generate report for {selected_month} {selected_year}.")

    elif view_type == "Expense vs Income":
        plot_expense_vs_income(selected_year)


# Savings Page
elif page == "Savings":
    st.header("Monthly Savings")

    # Input savings goal
    goal = st.number_input("Set Your Monthly Savings Goal (Tk):", min_value=0.0, value=st.session_state.savings_goal)
    st.session_state.savings_goal = goal

    # Display savings graph
    plot_savings_graph()

    savings_data = calculate_savings()
    if savings_data is not None and not savings_data.empty:
        st.subheader("Savings by Month")
        st.dataframe(savings_data[['Income', 'Expenses', 'Savings']])

        # Show progress for latest month
        last_savings = savings_data['Savings'].iloc[-1]
        st.write(f"Your savings this month: **{last_savings:.2f} Tk**")
        if goal > 0:
            progress = min(last_savings / goal, 1.0)
            st.progress(progress)
            st.write(f"Progress towards your goal: **{progress * 100:.1f}%**")

            # Share progress text
            share_text = f"I'm saving {last_savings:.2f} Tk this month, reaching {progress*100:.1f}% of my goal of {goal:.2f} Tk! 💰 #SavingsGoal"
            st.text_area("Copy and Share Your Savings Progress:", share_text, height=100)
        else:
            st.info("Set a savings goal above to track your progress.")

        # Show achievements badges
        badges = check_achievements()
        if badges:
            st.subheader("Achievements 🏆")
            for badge in badges:
                st.write(badge)
        else:
            st.write("No achievements yet. Keep saving!")
    else:
        st.info("No savings data available yet.")

# Balance Page
elif page == "Balance":
    st.header("Check Balance")

    # Button to calculate and display total and current balance
    if st.button("Calculate Balance"):
        total_balance, current_balance = calculate_balance()
        
        if total_balance is not None and current_balance is not None:
            st.subheader("Balance Details")
            st.write(f"**Total Balance:** {total_balance:.2f}Tk")
            st.write(f"**Current Balance (Latest Savings):** {current_balance:.2f}TK")
        else:
            st.warning("Insufficient data to calculate balances.")


# Expense Prediction Page
elif page == "Expense Prediction":
    st.header("Expense Prediction")

    # Predict Future Expenses
    months_ahead = st.number_input("Months Ahead", min_value=1, max_value=12, value=3)
    if st.button("Predict Expenses"):
        predictions = predict_expenses(months_ahead)
        if predictions is not None:
            st.subheader(f"Predicted Expenses for the Next {months_ahead} Months:")
            for i, prediction in enumerate(predictions):
                st.write(f"Month {i+1}: {prediction:.2f}TK")

# Display Footer
st.write("---")
st.markdown("This app allows you to track your income and expenses. Analyze your spending habits, visualize savings, and make predictions for future expenses!")
