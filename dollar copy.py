from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from functools import wraps
from bson import ObjectId
import pandas as pd
import os
from flask import jsonify
from datetime import datetime
import calendar
import plotly.express as px
import plotly.graph_objs as go
import re
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from prophet import Prophet
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key
app.config["MONGO_URI"] = "mongodb://localhost:27017/mercedes_showroom"
mongo = PyMongo(app)

# Create admin user if not exists
def create_admin():
    admin = mongo.db.users.find_one({"username": "ADMIN"})
    if not admin:
        admin_user = {
            "username": "ADMIN",
            "password": generate_password_hash("ADMIN123"),
            "role": "admin",
            "created_at": datetime.now()
        }
        mongo.db.users.insert_one(admin_user)

# Decorators for route protection
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session or session['role'] != 'admin':
            flash('Admin access required')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = mongo.db.users.find_one({"username": username})
        
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            session['role'] = user['role']
            session['user_id'] = str(user['_id'])
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    return render_template('login.html')

@app.route('/public_register', methods=['GET', 'POST'])
def public_register():
    if request.method == 'POST':
        # Check if username already exists
        existing_user = mongo.db.users.find_one({"username": request.form['username']})
        if existing_user:
            flash('Username already exists')
            return redirect(url_for('public_register'))

        try:
            # Calculate age from DOB
            dob = datetime.strptime(request.form['dob'], '%Y-%m-%d')
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            
            # Generate employee ID
            year = str(datetime.now().year)
            count = mongo.db.users.count_documents({"role": "sales_person"}) + 1
            employee_id = f"SP_{year}_{str(count).zfill(3)}"
            
            new_user = {
                "username": request.form['username'],
                "password": generate_password_hash(request.form['password']),
                "role": "sales_person",
                "employee_id": employee_id,
                "first_name": request.form['first_name'],
                "last_name": request.form['last_name'],
                "gender": request.form['gender'],
                "phone_number": request.form['phone_number'],
                "hire_date": datetime.now(),
                "salary": 0,  # Will be set by admin
                "work_schedule": "TBD",  # Will be set by admin
                "dob": dob,
                "age": age,
                "sales_month": 0,
                "sales_year": 0,
                "status": "pending",  # New users need admin approval
                "created_at": datetime.now()
            }
            
            mongo.db.users.insert_one(new_user)
            flash('Registration successful! Please wait for admin approval.')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error during registration: {str(e)}')
            return redirect(url_for('public_register'))
            
    return render_template('public_register.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if session['role'] == 'admin':
        # Admin Dashboard Data
        cars = list(mongo.db.cars.find())
        sales = list(mongo.db.sales.find())
        staff = list(mongo.db.users.find({"role": "sales_person"}))
        orders = list(mongo.db.orders.find())
        
        # Calculate statistics
        total_sales = len(sales)
        total_revenue = sum(sale.get('price', 0) for sale in sales)
        pending_approvals = mongo.db.users.count_documents({"role": "sales_person", "status": "pending"})
        pending_orders = mongo.db.orders.count_documents({"status": "pending"})
        
        # Monthly sales statistics
        current_month = datetime.now().month
        current_year = datetime.now().year
        monthly_sales = mongo.db.sales.count_documents({
            "sale_date": {
                "$gte": datetime(current_year, current_month, 1),
                "$lt": datetime(current_year, current_month + 1, 1) if current_month < 12 else datetime(current_year + 1, 1, 1)
            }
        })
        
        return render_template('admin_dashboard.html',
                             cars=cars,
                             sales=sales,
                             staff=staff,
                             orders=orders,
                             total_sales=total_sales,
                             total_revenue=total_revenue,
                             monthly_sales=monthly_sales,
                             pending_approvals=pending_approvals,
                             pending_orders=pending_orders)
    else:
        # Sales Person Dashboard Data
        user = mongo.db.users.find_one({"_id": ObjectId(session['user_id'])})
        available_cars = list(mongo.db.cars.find())
        user_orders = list(mongo.db.orders.find({"sales_person_id": session['user_id']}).sort("order_date", -1))
        
        # Calculate statistics
        total_sales = len([order for order in user_orders if order['status'] == 'approved'])
        total_revenue = sum(order.get('price', 0) for order in user_orders if order['status'] == 'approved')
        
        # Get recent orders
        recent_orders = user_orders[:5]
        
        return render_template('sales_dashboard.html',
                             user=user,
                             available_cars=available_cars,
                             recent_orders=recent_orders,
                             total_sales=total_sales,
                             total_revenue=total_revenue,
                             monthly_sales=len([order for order in user_orders 
                                              if order['status'] == 'approved' and 
                                              order['order_date'].month == datetime.now().month]))

@app.route('/place_order', methods=['POST'])
@login_required
def place_order():
    if request.method == 'POST':
        car_id = request.form['car_id']
        customer_name = request.form['customer_name']
        customer_phone = request.form['customer_phone']
        customer_email = request.form['customer_email']
        
        car = mongo.db.cars.find_one({"_id": ObjectId(car_id)})
        if not car:
            flash('Car not found')
            return redirect(url_for('dashboard'))
        
        order = {
            "car_id": car_id,
            "car_model": car['Model Name'],
            "price": car['Price of Model ($)'],
            "sales_person_id": session['user_id'],
            "sales_person_name": session['user'],
            "customer_name": customer_name,
            "customer_phone": customer_phone,
            "customer_email": customer_email,
            "order_date": datetime.now(),
            "status": "pending"
        }
        
        mongo.db.orders.insert_one(order)
        flash('Order placed successfully!')
        return redirect(url_for('dashboard'))

@app.route('/orders')
@admin_required
def orders():
    all_orders = list(mongo.db.orders.find().sort("order_date", -1))
    return render_template('orders.html', orders=all_orders)

@app.route('/update_order_status/<order_id>/<status>')
@admin_required
def update_order_status(order_id, status):
    try:
        mongo.db.orders.update_one(
            {"_id": ObjectId(order_id)},
            {"$set": {"status": status}}
        )
        
        # If order is approved, update sales statistics
        if status == 'approved':
            order = mongo.db.orders.find_one({"_id": ObjectId(order_id)})
            if order:
                # Add to sales collection
                sale = {
                    "order_id": order_id,
                    "car_id": order['car_id'],
                    "car_model": order['car_model'],
                    "price": order['price'],
                    "sales_person_id": order['sales_person_id'],
                    "sales_person_name": order['sales_person_name'],
                    "customer_name": order['customer_name'],
                    "sale_date": datetime.now()
                }
                mongo.db.sales.insert_one(sale)
                
                # Update sales person's statistics
                mongo.db.users.update_one(
                    {"_id": ObjectId(order['sales_person_id'])},
                    {
                        "$inc": {
                            "sales_month": 1,
                            "sales_year": 1
                        }
                    }
                )
        
        flash(f'Order {status} successfully!')
    except Exception as e:
        flash(f'Error updating order status: {str(e)}')
    return redirect(url_for('orders'))

@app.route('/manage_staff')
@admin_required
def manage_staff():
    staff = list(mongo.db.users.find({"role": "sales_person"}))
    return render_template('manage_staff.html', staff=staff)

@app.route('/approve_sales_person/<user_id>')
@admin_required
def approve_sales_person(user_id):
    try:
        mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"status": "active"}}
        )
        flash('Sales person approved successfully!')
    except Exception as e:
        flash(f'Error approving sales person: {str(e)}')
    return redirect(url_for('manage_staff'))

@app.route('/reject_sales_person/<user_id>')
@admin_required
def reject_sales_person(user_id):
    try:
        mongo.db.users.delete_one({"_id": ObjectId(user_id)})
        flash('Sales person rejected and removed from system.')
    except Exception as e:
        flash(f'Error rejecting sales person: {str(e)}')
    return redirect(url_for('manage_staff'))

@app.route('/update_sales_person/<user_id>', methods=['POST'])
@admin_required
def update_sales_person(user_id):
    try:
        updates = {
            "salary": float(request.form['salary']),
            "work_schedule": request.form['work_schedule']
        }
        
        mongo.db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": updates}
        )
        flash('Sales person information updated successfully!')
    except Exception as e:
        flash(f'Error updating sales person information: {str(e)}')
    return redirect(url_for('manage_staff'))

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('login'))

# Car Management Routes
@app.route('/add_car', methods=['POST'])
@admin_required
def add_car():
    try:
        new_car = {
            "Model Name": request.form['model_name'],
            "Type": request.form['type'],
            "Year": int(request.form['year']),
            "Price of Model ($)": float(request.form['price']),
            "created_at": datetime.now()
        }
        
        mongo.db.cars.insert_one(new_car)
        flash('Car added successfully!')
    except Exception as e:
        flash(f'Error adding car: {str(e)}')
    return redirect(url_for('dashboard'))

@app.route('/update_car/<car_id>', methods=['POST'])
@admin_required
def update_car(car_id):
    try:
        updates = {
            "Model Name": request.form['model_name'],
            "Type": request.form['type'],
            "Year": int(request.form['year']),
            "Price of Model ($)": float(request.form['price'])
        }
        
        mongo.db.cars.update_one(
            {"_id": ObjectId(car_id)},
            {"$set": updates}
        )
        flash('Car information updated successfully!')
    except Exception as e:
        flash(f'Error updating car information: {str(e)}')
    return redirect(url_for('dashboard'))
from flask import Flask, render_template
import pandas as pd
import json
from datetime import datetime
import calendar
from bson import json_util

@app.route('/sales_analysis')
@admin_required
def sales_analysis():
    # Fetch data from MongoDB
    current_year_data = list(mongo.db.cars.find())
    last_year_data = list(mongo.db.last_year_sales_data.find())
    
    # 1. Monthly Sales Analysis
    def get_monthly_sales(data):
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_totals = []
        
        for month in months:
            total = 0
            for record in data:
                # Handle both regular and space-suffixed month names
                month_val = record.get(month, record.get(f'{month} ', 0))
                if isinstance(month_val, str):
                    month_val = float(month_val) if month_val.replace('.', '').isdigit() else 0
                elif isinstance(month_val, (int, float)) and not pd.isna(month_val):
                    month_val = float(month_val)
                else:
                    month_val = 0
                total += month_val
            monthly_totals.append(total)
        
        return monthly_totals

    current_monthly_sales = get_monthly_sales(current_year_data)
    last_monthly_sales = get_monthly_sales(last_year_data)
    
    # 2. Model-wise Analysis
    model_analysis = []
    for current_model in current_year_data:
        model_name = current_model['Model Name']
        last_model = next((m for m in last_year_data if m['Model Name'] == model_name), None)
        
        # Get total sales for current year
        current_total = sum(float(current_model.get(m, current_model.get(f'{m} ', 0)) or 0) 
                          for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        # Get total sales for last year
        last_total = 0
        if last_model:
            last_total = float(last_model.get('Total ', 0))
        
        # Calculate growth
        growth = ((current_total - last_total) / last_total * 100) if last_total > 0 else 100
        
        model_analysis.append({
            'model': model_name,
            'current_sales': current_total,
            'last_year_sales': last_total,
            'growth': growth,
            'price': current_model['Price of Model ($)'],
            'revenue': current_total * current_model['Price of Model ($)']
        })
    
    # 3. Car Type Analysis
    car_types = {}
    for car in current_year_data:
        car_type = car['Type of Car']
        car_types[car_type] = car_types.get(car_type, 0) + 1
    
    # 4. Price Segment Analysis
    def get_price_segment(price):
        if price < 7500000:
            return 'Economy (< 75L)'
        elif price < 14000000:
            return 'Mid-range (<1.4 Cr)'
        elif price < 25000000:
            return 'Luxury (< 2Cr)'
        else:
            return 'Ultra - Luxury (> 2Cr)'
    
    price_segments = {}
    for car in current_year_data:
        segment = get_price_segment(car['Price of Model ($)'])
        price_segments[segment] = price_segments.get(segment, 0) + 1
    
    # 5. Technical Specifications Analysis
    fuel_types = {}
    engine_types = {}
    total_mileage = 0
    total_airbags = 0
    count = 0
    
    for car in current_year_data:
        # Fuel types
        fuel_type = car['Type of Fuel']
        fuel_types[fuel_type] = fuel_types.get(fuel_type, 0) + 1
        
        # Engine types
        engine_type = car['Engine Type']
        engine_types[engine_type] = engine_types.get(engine_type, 0) + 1
        
        # Mileage calculation
        try:
            mileage = float(car['Mileage / Range'].split()[0])
            total_mileage += mileage
            count += 1
        except (ValueError, AttributeError, KeyError):
            pass
        
        # Airbags
        total_airbags += car.get('Number of Airbags', 0)
    
    tech_specs = {
        'fuel_types': fuel_types,
        'engine_types': engine_types,
        'avg_mileage': total_mileage / count if count > 0 else 0,
        'avg_airbags': total_airbags / len(current_year_data) if current_year_data else 0
    }
    
    # 6. Calculate total revenue and growth
    total_revenue_current = sum(model['revenue'] for model in model_analysis)
    total_revenue_last = sum(float(car.get('Total ', 0)) * car['Price of Model ($)'] 
                           for car in last_year_data)
    
    revenue_growth = ((total_revenue_current - total_revenue_last) / total_revenue_last * 100) \
                    if total_revenue_last > 0 else 100
    
    # Prepare data for charts
    monthly_sales_chart = {
        'labels': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        'current_year': current_monthly_sales,
        'last_year': last_monthly_sales
    }
    
    # Sort model analysis by revenue
    model_analysis_sorted = sorted(model_analysis, key=lambda x: x['revenue'], reverse=True)
    
    analysis_data = {
        'monthly_sales_chart': json.dumps(monthly_sales_chart),
        'model_analysis': model_analysis_sorted,
        'car_types': json.dumps(car_types),
        'price_segments': json.dumps(price_segments),
        'tech_specs': tech_specs,
        'total_revenue_current': total_revenue_current,
        'total_revenue_last': total_revenue_last,
        'revenue_growth': revenue_growth,
        'total_units_current': sum(current_monthly_sales),
        'total_units_last': sum(last_monthly_sales),
        'units_growth': ((sum(current_monthly_sales) - sum(last_monthly_sales)) / sum(last_monthly_sales) * 100) 
                        if sum(last_monthly_sales) > 0 else 100
    }
    
    return render_template('sales_analysis.html', **analysis_data)

@app.route('/delete_car/<car_id>')
@admin_required
def delete_car(car_id):
    try:
        mongo.db.cars.delete_one({"_id": ObjectId(car_id)})
        flash('Car removed successfully!')
    except Exception as e:
        flash(f'Error removing car: {str(e)}')
    return redirect(url_for('dashboard'))
@app.route('/mercedes-sales')
def mercedes_sales():
    # Load the data
    df = pd.read_csv("Mercedes_sales_data_with_models.csv")
    
    # Print column names for debugging
    print("Columns in the dataset:", df.columns.tolist())
    
    # Data Cleaning and Preparation
    df.columns = [re.sub('[ -]', '_', c).lower().strip() for c in df.columns]  # Clean column names
    df['price_of_model'] = df['price_of_model'].replace('[\$,]', '', regex=True).astype(float)  # Clean price column

    # Extract horsepower and torque from 'power_/_torque'
    df['horsepower'] = df['power_/_torque'].apply(
        lambda x: float(re.search(r'\d+', x.split('/')[0]).group()) if pd.notna(x) and isinstance(x, str) else None
    )
    df['torque'] = df['power_/_torque'].apply(
        lambda x: float(re.search(r'\d+', x.split('/')[1]).group()) if pd.notna(x) and isinstance(x, str) else None
    )

    # Drop rows with missing horsepower or torque values (optional)
    df = df.dropna(subset=['horsepower', 'torque'])

    # Convert monthly sales columns to numeric
    for month in ['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sep', 'oct', 'nov', 'dec']:
        df[month] = pd.to_numeric(df[month], errors='coerce')  # Convert to numeric, invalid values become NaN

    # Analysis
    # 1. Total Sales by Model
    total_sales_by_model = df.groupby('model_name')['total'].sum().reset_index()

    # 2. Average Price by Car Type
    avg_price_by_type = df.groupby('type_of_car')['price_of_model'].mean().reset_index()

    # 3. Monthly Sales Trends
    monthly_sales = df[['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sep', 'oct', 'nov', 'dec']].sum().reset_index()
    monthly_sales.columns = ['month', 'sales']

    # 4. Sales by Fuel Type
    sales_by_fuel = df.groupby('type_of_fuel')['total'].sum().reset_index()

    # 5. Top 5 Models by Total Sales
    top_5_models = df.groupby('model_name')['total'].sum().nlargest(5).reset_index()

    # 6. Correlation between Price and Horsepower
    if not df.empty and 'price_of_model' in df.columns and 'horsepower' in df.columns:
        price_horsepower_corr = df[['price_of_model', 'horsepower']].corr().iloc[0, 1]
    else:
        price_horsepower_corr = None  # Handle invalid data

    # Print correlation for debugging
    print("Correlation between Price and Horsepower:", price_horsepower_corr)

    # Create 2D Plotly Graphs
    def create_bar_plot(data, x, y, title, xaxis_title, yaxis_title):
        fig = px.bar(data, x=x, y=y, title=title, labels={x: xaxis_title, y: yaxis_title})
        return fig.to_html(full_html=False)

    def create_line_plot(data, x, y, title, xaxis_title, yaxis_title):
        fig = px.line(data, x=x, y=y, title=title, labels={x: xaxis_title, y: yaxis_title})
        return fig.to_html(full_html=False)

    def create_pie_chart(data, names, values, title):
        fig = px.pie(data, names=names, values=values, title=title)
        return fig.to_html(full_html=False)

    def create_scatter_plot(data, x, y, title, xaxis_title, yaxis_title):
        fig = px.scatter(data, x=x, y=y, title=title, labels={x: xaxis_title, y: yaxis_title})
        return fig.to_html(full_html=False)

    # Plot 1: Total Sales by Model
    plot1 = create_bar_plot(total_sales_by_model, 'model_name', 'total', 'Total Sales by Model', 'Model Name', 'Total Sales')

    # Plot 2: Average Price by Car Type
    plot2 = create_bar_plot(avg_price_by_type, 'type_of_car', 'price_of_model', 'Average Price by Car Type', 'Car Type', 'Average Price')

    # Plot 3: Monthly Sales Trends
    plot3 = create_line_plot(monthly_sales, 'month', 'sales', 'Monthly Sales Trends', 'Month', 'Sales')

    # Plot 4: Sales by Fuel Type
    plot4 = create_pie_chart(sales_by_fuel, 'type_of_fuel', 'total', 'Sales by Fuel Type')

    # Plot 5: Top 5 Models by Total Sales
    plot5 = create_bar_plot(top_5_models, 'model_name', 'total', 'Top 5 Models by Total Sales', 'Model Name', 'Total Sales')

    # Plot 6: Scatter Plot for Price vs Horsepower
    plot6 = create_scatter_plot(df, 'horsepower', 'price_of_model', 'Price vs Horsepower', 'Horsepower', 'Price of Model')

    # Render the template with the plots and data
    return render_template('mercedes_sales.html', 
                           plot1=plot1, 
                           plot2=plot2, 
                           plot3=plot3, 
                           plot4=plot4, 
                           plot5=plot5, 
                           plot6=plot6, 
                           price_horsepower_corr=price_horsepower_corr)


@app.route('/car-sales')
def car_sales():
    # Load the data
    df = pd.read_csv("car_sales_kaggle.csv")
    
    # Data Cleaning and Preparation
    df = df.drop(columns=['Phone'])
    df.columns = [re.sub('[ -]', '_', c).lower().strip() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['year_month'] = df['date'].astype(str).str[:7]
    
    # Analysis
    car_sales_by_year_month = df.groupby(['year_month']).agg(number_car=('car_id', 'nunique')).sort_index().reset_index()
    order_revenue = df.groupby(['company']).agg(total_order=('car_id', 'count'), total_revenue=('price_($)', 'sum')).sort_values(by='total_order', ascending=False).reset_index()
    body_car_sales = df.groupby(['body_style', 'year']).agg(number_order=('car_id', 'count'), revenue=('price_($)', 'sum')).reset_index()
    top_5_model = df.model.value_counts().nlargest(5).reset_index()  # Convert Series to DataFrame
    transmission_count = df.transmission.value_counts()
    transmission_percent = (transmission_count / transmission_count.sum() * 100).reset_index()  # Convert Series to DataFrame
    gender_value = df.gender.value_counts().reset_index()  # Convert Series to DataFrame
    gender_value.columns = ['gender', 'count']  # Rename columns for clarity
    gender_data = df.groupby(['gender', 'body_style']).size().reset_index(name='Sale')
    
    # Create 2D Plotly Graphs
    def create_bar_plot(data, x, y, title, xaxis_title, yaxis_title):
        fig = px.bar(data, x=x, y=y, title=title, labels={x: xaxis_title, y: yaxis_title})
        return fig.to_html(full_html=False)

    def create_line_plot(data, x, y, title, xaxis_title, yaxis_title):
        fig = px.line(data, x=x, y=y, title=title, labels={x: xaxis_title, y: yaxis_title})
        return fig.to_html(full_html=False)

    def create_scatter_plot(data, x, y, title, xaxis_title, yaxis_title):
        fig = px.scatter(data, x=x, y=y, title=title, labels={x: xaxis_title, y: yaxis_title})
        return fig.to_html(full_html=False)

    def create_pie_chart(data, names, values, title):
        fig = px.pie(data, names=names, values=values, title=title)
        return fig.to_html(full_html=False)

    # Plot 1: Bar Plot for Car Sales by Year-Month
    plot1 = create_bar_plot(car_sales_by_year_month, 'year_month', 'number_car', 'Car Sales by Year-Month', 'Year-Month', 'Number of Cars Sold')

    # Plot 2: Bar Plot for Total Order and Revenue by Company
    plot2 = create_bar_plot(order_revenue, 'company', 'total_order', 'Total Orders by Company', 'Company', 'Total Orders')

    # Plot 3: Line Plot for Order and Revenue by Body Style and Year
    plot3 = create_line_plot(body_car_sales, 'year', 'revenue', 'Revenue by Body Style and Year', 'Year', 'Revenue')

    # Plot 4: Scatter Plot for Price Distribution by Body Style
    plot4 = create_scatter_plot(df, 'body_style', 'price_($)', 'Price Distribution by Body Style', 'Body Style', 'Price ($)')

    # Plot 5: Pie Chart for Gender Distribution
    plot5 = create_pie_chart(gender_value, 'gender', 'count', 'Gender Distribution')

    # Render the template with the plots and data
    return render_template('car_sales.html', 
                           plot1=plot1, 
                           plot2=plot2, 
                           plot3=plot3, 
                           plot4=plot4, 
                           plot5=plot5, 
                           top_5_model=top_5_model, 
                           transmission_percent=transmission_percent, 
                           gender_value=gender_value)


import os
from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# ======== API KEYS ========
GROQ_API_KEY = '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'
GOOGLE_API_KEY = 'AIzaSyBkoorRTaH08H3RFIft4ug6bT1ABexXswI'
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class FinancialAIChatbot:
    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="Llama3-8b-8192",
            temperature=0.3
        )

        self.rag_prompt = ChatPromptTemplate.from_template("""
            As a sales and business expert, analyze the given data and provide actionable insights based on:
            1. Monthly, quarterly, and yearly sales trends.
            2. Strategic opportunities to boost sales and profits.
            3. Industry benchmarks and competitive analysis.
            
            Context: {context}
            Question: {input}

            Your response should include:
            - Sales performance analysis and trends.
            - Practical recommendations for improving sales (e.g., promotions, pricing, or bundling strategies).
            - Insights on underperforming areas and corrective measures.
            - Opportunities for maximizing profits, such as targeting high-demand products.

            Answer concisely in one well-structured paragraph, focusing on actionable suggestions:
        """)

        self.retrieval_chain = self.setup_retrieval_chain()

    def setup_retrieval_chain(self):
        try:
            if not os.path.exists("financial_docs"):
                raise FileNotFoundError("Document directory not found")
            
            loader = PyPDFDirectoryLoader("financial_docs")
            docs = loader.load()
            
            if not docs:
                raise ValueError("No documents loaded - check PDF files")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1800,
                chunk_overlap=400,
                separators=["\n\nSection:", "\n\nArticle:", "\n\n"]
            )
            documents = text_splitter.split_documents(docs)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            test_embed = embeddings.embed_query("test")
            if not isinstance(test_embed, list) or len(test_embed) < 1:
                raise ValueError("Embedding initialization failed")
            
            vector_store = FAISS.from_documents(documents, embeddings)
            return create_retrieval_chain(
                vector_store.as_retriever(search_kwargs={"k": 5}),
                create_stuff_documents_chain(self.llm, self.rag_prompt)
            )
        except Exception as e:
            print(f"System initialization failed: {str(e)}")
            exit()

    def get_response(self, user_input):
        try:
            result = self.retrieval_chain.invoke({"input": user_input})
            return self._format_answer(result.get("answer", ""))
        except Exception as e:
            return f"Response error: {str(e)}"

    def _format_answer(self, answer):
        return f"{answer.strip()}\n\n[Sources: Provided documents + Sales database]"

# Initialize chatbot instance
chatbot = FinancialAIChatbot()


@app.route("/chat")
@login_required
def chat():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
@login_required
def ask():
    try:
        data = request.json
        user_input = data.get("question", "")
        if not user_input:
            return jsonify({"error": "No question provided"}), 400
        
        response = chatbot.get_response(user_input)
        return jsonify({
            "response": response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

def prepare_features(data):
    """Prepare features for time series prediction"""
    features = pd.DataFrame({
        'month_num': range(1, 13),
        'is_quarter_end': [i % 3 == 0 for i in range(1, 13)],
        'is_year_end': [i == 12 for i in range(1, 13)]
    })
    
    # Add seasonal features
    features['sin_month'] = np.sin(2 * np.pi * features['month_num']/12)
    features['cos_month'] = np.cos(2 * np.pi * features['month_num']/12)
    
    return features

def prepare_future_features():
    """Prepare features for January 2025"""
    return pd.DataFrame({
        'month_num': [1],
        'is_quarter_end': [False],
        'is_year_end': [False],
        'sin_month': [np.sin(2 * np.pi * 1/12)],
        'cos_month': [np.cos(2 * np.pi * 1/12)]
    })

@app.route('/sales_predictions')
@admin_required
def sales_predictions():
    # Get all unique car models from the database
    df = pd.read_csv("Mercedes_sales_data_with_models.csv")
    car_models = df['Model Name'].unique().tolist()
    return render_template('sales_predictions.html', car_models=car_models)

@app.route('/get_predictions/<model_name>')
@admin_required
def get_predictions(model_name):
    try:
        # Load and prepare data
        df = pd.read_csv("Mercedes_sales_data_with_models.csv")
        car_data = df[df['Model Name'] == model_name].iloc[0]
        
        # Prepare features
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Convert sales data to numeric with proper error handling
        y_train = []
        for month in months:
            value = car_data[month]
            if pd.isna(value):
                # Use mean of surrounding months instead of 0
                surrounding_values = [
                    float(car_data[m]) for m in months 
                    if pd.notna(car_data[m])
                ]
                if surrounding_values:
                    y_train.append(np.mean(surrounding_values))
                else:
                    y_train.append(0)
            else:
                try:
                    y_train.append(float(value))
                except ValueError:
                    # Try cleaning the string and converting again
                    cleaned_value = str(value).replace(',', '').strip()
                    try:
                        y_train.append(float(cleaned_value))
                    except ValueError:
                        y_train.append(np.mean(y_train) if y_train else 0)
        
        y_train = np.array(y_train)
        
        # Prepare training features
        X_train = prepare_features(y_train)
        X_pred = prepare_future_features()
        
        # Train models and make predictions with cross-validation
        # 1. Linear Regression with regularization
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = max(0, float(lr_model.predict(X_pred)[0]))
        
        # 2. Random Forest with more trees and deeper learning
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        rf_pred = max(0, float(rf_model.predict(X_pred)[0]))
        
        # 3. LightGBM with better parameters
        lgb_model = lgb.LGBMRegressor(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            random_state=42
        )
        lgb_model.fit(X_train, y_train)
        lgb_pred = max(0, float(lgb_model.predict(X_pred)[0]))
        
        # 4. Prophet with better seasonality handling
        prophet_df = pd.DataFrame({
            'ds': [datetime.strptime(f"2024-{i:02d}-01", "%Y-%m-%d") for i in range(1, 13)],
            'y': y_train
        })
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        prophet_model.fit(prophet_df)
        future_dates = pd.DataFrame({
            'ds': [datetime.strptime("2025-01-01", "%Y-%m-%d")]
        })
        prophet_pred = max(0, float(prophet_model.predict(future_dates)['yhat'].values[0]))
        
        # Calculate confidence in predictions based on historical variance
        historical_std = np.std(y_train)
        historical_mean = np.mean(y_train)
        confidence_factor = 1 - (historical_std / (historical_mean + 1e-6))  # Avoid division by zero
        
        # Adjust predictions using confidence and historical data
        predictions = {
            'linear': lr_pred,
            'random_forest': rf_pred,
            'lightgbm': lgb_pred,
            'prophet': prophet_pred
        }
        
        # Ensemble prediction with confidence weighting
        weighted_pred = (
            confidence_factor * np.mean(list(predictions.values())) +
            (1 - confidence_factor) * historical_mean
        )
        
        # Prepare response data
        response = {
            'predictions': {
                'linear': lr_pred,
                'random_forest': rf_pred,
                'lightgbm': lgb_pred,
                'prophet': prophet_pred,
                'ensemble': weighted_pred
            },
            'historical_data': {
                'months': months,
                'sales': y_train.tolist()
            },
            'car_details': {
                'type': car_data['Type of Car'],
                'price': float(str(car_data['Price of Model']).replace(',', '')),
                'avg_sales': float(np.mean(y_train)),
                'prediction_confidence': round(confidence_factor * 100, 2)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")  # Add logging
        return jsonify({'error': str(e)}), 500

# app.py
import os
from pathlib import Path
import fitz  # PyMuPDF
from flask import Flask, render_template
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from langchain_groq import ChatGroq
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime

# Initialize Flask app
# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Data directory setup
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# API keys for ChatGroq (example keys provided)
API_KEYS = [
    "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",
    "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",
    "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",
    "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",
    "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",
]

# Create ChatGroq instances
llm_instances = [
    ChatGroq(groq_api_key=key, model_name="Llama3-8b-8192", temperature=0.3)
    for key in API_KEYS
]

llm_lock = threading.Lock()
current_llm_index = 0

def analyze_sentiment(text):
    """
    Analyze sentiment of text using NLTK's VADER sentiment analyzer.
    Returns sentiment label and score.
    """
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']
    
    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return {
        'sentiment': sentiment,
        'sentiment_score': compound_score,
        'details': scores
    }

def get_next_llm():
    """Return the next ChatGroq instance in round-robin fashion."""
    global current_llm_index
    with llm_lock:
        instance = llm_instances[current_llm_index]
        current_llm_index = (current_llm_index + 1) % len(llm_instances)
    return instance

def extract_reviews(pdf_path):
    """Extract text from a PDF file."""
    try:
        with fitz.open(pdf_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text("text")
        return text
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return None

def extract_review_details(section, current_model):
    """Extract and structure review details from a section of text."""
    lines = section.splitlines()
    details = {
        "model": current_model,
        "date": datetime.now().strftime("%Y-%m-%d"),  # Default date
        "rating": "Not Specified",
        "text": section.strip()
    }
    
    for line in lines:
        if "Model:" in line:
            parts = line.split("Model:")
            if len(parts) > 1:
                details["model"] = parts[1].strip()
                current_model = details["model"]
        elif "Date:" in line:
            parts = line.split("Date:")
            if len(parts) > 1:
                details["date"] = parts[1].strip()
        elif "Rating:" in line:
            parts = line.split("Rating:")
            if len(parts) > 1:
                details["rating"] = parts[1].strip()
    
    return details, current_model

def get_ai_summary_for_model(model, combined_text):
    """
    Generate an AI summary for reviews of a given model.
    The summary is requested in bullet points with each bullet a single concise sentence.
    """
    prompt = f"""
    Provide a summary of the reviews for the {model} in bullet points.
    Each bullet point should be a single concise sentence that highlights:
      - Key positive aspects,
      - Key negative aspects, or
      - A suggested improvement.
    Do not include any introductory or trailing text.
    Reviews:
    {combined_text}
    """
    
    try:
        llm_instance = get_next_llm()
        response = llm_instance.invoke(prompt)
        # Return the AI response as text (assumed to contain bullet points)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"AI summary unavailable: {str(e)}"

# Custom Jinja filter to convert summary text to bullet points
def convert_summary_to_bullets(summary):
    """
    Converts a plain text summary into HTML bullet points.
    Lines that start with an asterisk (*) become list items.
    Other lines are wrapped in <p>.
    """
    if not summary:
        return ""
    lines = summary.splitlines()
    output = ""
    in_list = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("*"):
            if not in_list:
                output += "<ul>"
                in_list = True
            # Remove the asterisk and any extra space
            bullet = line.lstrip("*").strip()
            output += f"<li>{bullet}</li>"
        else:
            if in_list:
                output += "</ul>"
                in_list = False
            output += f"<p>{line}</p>"
    if in_list:
        output += "</ul>"
    return output

# Register the custom filter
app.jinja_env.filters['bullet_summary'] = convert_summary_to_bullets

@app.route("/customer_review_analysis")
def customer_review_analysis():
    """Main route for customer review analysis."""
    pdf_path = DATA_DIR / "reviews.pdf"
    
    if not pdf_path.exists():
        return render_template(
            "customer_review_analysis.html",
            error="PDF file not found in data directory",
            show_results=False
        )

    text = extract_reviews(pdf_path)
    if not text:
        return render_template(
            "customer_review_analysis.html",
            error="Error processing PDF file",
            show_results=False
        )

    # Process reviews by splitting on a marker (e.g., "Reviewer:")
    reviews = []
    sections = text.split("Reviewer:")
    current_model = "Unknown Model"
    
    for section in sections[1:]:  # Skip the first (possibly empty) section
        details, current_model = extract_review_details(section, current_model)
        # Analyze sentiment for each review text
        sentiment_data = analyze_sentiment(details["text"])
        reviews.append({
            "details": details,
            "sentiment": sentiment_data
        })

    # Calculate overall statistics
    total_reviews = len(reviews)
    positive_reviews = sum(1 for r in reviews if r["sentiment"]["sentiment"] == "Positive")
    negative_reviews = sum(1 for r in reviews if r["sentiment"]["sentiment"] == "Negative")
    neutral_reviews = total_reviews - positive_reviews - negative_reviews
    
    satisfaction_rate = (positive_reviews / total_reviews * 100) if total_reviews > 0 else 0

    # Group reviews by model for AI summarization
    model_reviews = {}
    for review in reviews:
        model = review["details"]["model"]
        if model not in model_reviews:
            model_reviews[model] = []
        model_reviews[model].append(review["details"]["text"])

    # Generate summaries for each model concurrently
    model_summaries = {}
    with ThreadPoolExecutor() as executor:
        future_summaries = {
            model: executor.submit(get_ai_summary_for_model, model, "\n\n".join(texts))
            for model, texts in model_reviews.items()
        }
        model_summaries = {
            model: future.result()
            for model, future in future_summaries.items()
        }

    return render_template(
        "customer_review_analysis.html",
        reviews=reviews,
        model_summaries=model_summaries,
        total_reviews=total_reviews,
        positive_count=positive_reviews,
        negative_count=negative_reviews,
        satisfaction_rate=round(satisfaction_rate, 1),
        show_results=True,
        error=None
    )

if __name__ == '__main__':
    app.run(debug=True)
