"""
Advanced Demo Data Generator for Personal Finance Dashboard
Perfect for software engineering internship interviews
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid

class FinanceDataGenerator:
    def __init__(self):
        # Realistic merchant categories
        self.merchants = {
            'grocery': ['Whole Foods', 'Trader Joes', 'Kroger', 'Safeway', 'Walmart Grocery'],
            'retail': ['Amazon', 'Target', 'Walmart', 'Costco', 'Best Buy', 'Apple Store'],
            'food': ['Starbucks', 'McDonalds', 'Chipotle', 'Subway', 'Pizza Hut', 'KFC'],
            'gas': ['Shell', 'Exxon', 'Chevron', 'BP', 'Mobil', '76'],
            'entertainment': ['Netflix', 'Spotify', 'AMC Theaters', 'Dave & Busters'],
            'healthcare': ['CVS Pharmacy', 'Walgreens', 'Kaiser Permanente', 'Urgent Care'],
            'transportation': ['Uber', 'Lyft', 'Metro Transit', 'Parking Meter'],
            'utilities': ['PG&E', 'Comcast', 'Verizon', 'AT&T'],
            'fitness': ['24 Hour Fitness', 'Planet Fitness', 'Yoga Studio'],
            'other': ['Home Depot', 'Lowes', 'Bank Fee', 'ATM Fee']
        }
        
        # Spending patterns by category (average amounts)
        self.category_patterns = {
            'grocery': {'min': 15, 'max': 200, 'avg': 75, 'frequency': 0.8},
            'retail': {'min': 25, 'max': 500, 'avg': 120, 'frequency': 0.4},
            'food': {'min': 8, 'max': 60, 'avg': 25, 'frequency': 0.6},
            'gas': {'min': 30, 'max': 80, 'avg': 45, 'frequency': 0.3},
            'entertainment': {'min': 12, 'max': 100, 'avg': 35, 'frequency': 0.2},
            'healthcare': {'min': 20, 'max': 300, 'avg': 85, 'frequency': 0.1},
            'transportation': {'min': 8, 'max': 45, 'avg': 18, 'frequency': 0.4},
            'utilities': {'min': 50, 'max': 250, 'avg': 120, 'frequency': 0.05},
            'fitness': {'min': 30, 'max': 150, 'avg': 65, 'frequency': 0.15},
            'other': {'min': 10, 'max': 400, 'avg': 95, 'frequency': 0.3}
        }
        
        # Cities and states for realistic geography
        self.locations = [
            ('San Francisco', 'CA'), ('New York', 'NY'), ('Seattle', 'WA'),
            ('Austin', 'TX'), ('Chicago', 'IL'), ('Boston', 'MA'),
            ('Los Angeles', 'CA'), ('Denver', 'CO'), ('Miami', 'FL'),
            ('Portland', 'OR')
        ]
        
        # Channels
        self.channels = ['online', 'in-store', 'mobile', 'phone']
    
    def generate_realistic_transactions(self, num_transactions=1000, days_back=90):
        """Generate realistic transaction data"""
        transactions = []
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Generate user profiles (different spending patterns)
        user_profiles = self._generate_user_profiles()
        
        for i in range(num_transactions):
            # Select random user profile
            profile = random.choice(user_profiles)
            
            # Generate timestamp with realistic patterns
            timestamp = self._generate_realistic_timestamp(start_date, end_date, profile)
            
            # Select category based on profile and time patterns
            category = self._select_category_by_time_and_profile(timestamp, profile)
            
            # Select merchant from category
            merchant = random.choice(self.merchants[category])
            
            # Generate amount based on category and profile
            amount = self._generate_amount(category, profile, timestamp)
            
            # Generate location
            city, state = random.choice(self.locations)
            
            # Generate other fields
            transaction = {
                'tx_id': str(uuid.uuid4()),
                'timestamp': timestamp,
                'amount': round(amount, 2),
                'currency': 'USD',
                'merchant': merchant,
                'mcc': self._get_mcc_code(category),
                'account_id': profile['account_id'],
                'city': city,
                'state': state,
                'channel': self._select_channel(category, timestamp),
                'category': category,
                'subcategory': self._get_subcategory(category, merchant),
                'is_recurring': self._is_recurring_transaction(merchant, category),
                'normalized_amount': round(amount, 2)
            }
            
            transactions.append(transaction)
        
        # Convert to DataFrame and sort by timestamp
        df = pd.DataFrame(transactions)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _generate_user_profiles(self):
        """Generate different user spending profiles"""
        profiles = [
            {
                'account_id': 'acc_frugal_user',
                'spending_level': 'low',
                'weekend_multiplier': 1.1,
                'categories_preference': ['grocery', 'utilities', 'gas'],
                'avg_daily_budget': 85
            },
            {
                'account_id': 'acc_moderate_user',
                'spending_level': 'medium',
                'weekend_multiplier': 1.4,
                'categories_preference': ['grocery', 'retail', 'food', 'entertainment'],
                'avg_daily_budget': 150
            },
            {
                'account_id': 'acc_high_spender',
                'spending_level': 'high',
                'weekend_multiplier': 1.8,
                'categories_preference': ['retail', 'food', 'entertainment', 'fitness'],
                'avg_daily_budget': 250
            }
        ]
        return profiles
    
    def _generate_realistic_timestamp(self, start_date, end_date, profile):
        """Generate timestamps with realistic patterns"""
        # Random date
        days_diff = (end_date - start_date).days
        random_days = random.randint(0, days_diff)
        base_date = start_date + timedelta(days=random_days)
        
        # Time patterns (more likely during business hours and evenings)
        hour_weights = {
            range(6, 9): 0.8,    # Morning
            range(9, 12): 1.2,   # Mid-morning
            range(12, 14): 1.5,  # Lunch
            range(14, 17): 1.0,  # Afternoon
            range(17, 20): 1.8,  # Evening (peak)
            range(20, 23): 1.3,  # Night
            range(23, 6): 0.2    # Late night/early morning
        }
        
        # Select hour based on weights
        hour_choices = []
        for hour_range, weight in hour_weights.items():
            hour_choices.extend([h for h in hour_range for _ in range(int(weight * 10))])
        
        hour = random.choice(hour_choices)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        timestamp = base_date.replace(hour=hour, minute=minute, second=second)
        return timestamp
    
    def _select_category_by_time_and_profile(self, timestamp, profile):
        """Select category based on time and user profile"""
        # Weekend vs weekday patterns
        is_weekend = timestamp.weekday() >= 5
        hour = timestamp.hour
        
        # Time-based category preferences
        if 6 <= hour <= 11:  # Morning
            morning_categories = ['grocery', 'food', 'gas']
        elif 11 <= hour <= 14:  # Lunch
            morning_categories = ['food', 'retail', 'grocery']
        elif 17 <= hour <= 22:  # Evening
            morning_categories = ['food', 'entertainment', 'retail', 'grocery']
        else:
            morning_categories = list(self.merchants.keys())
        
        # Weekend adjustments
        if is_weekend:
            weekend_categories = ['entertainment', 'retail', 'food', 'fitness']
            morning_categories.extend(weekend_categories)
        
        # Profile preferences
        preferred_categories = profile.get('categories_preference', list(self.merchants.keys()))
        
        # Combine all factors
        possible_categories = list(set(morning_categories) & set(preferred_categories))
        if not possible_categories:
            possible_categories = preferred_categories
        
        return random.choice(possible_categories)
    
    def _generate_amount(self, category, profile, timestamp):
        """Generate realistic transaction amount"""
        pattern = self.category_patterns[category]
        base_amount = random.uniform(pattern['min'], pattern['max'])
        
        # Profile adjustments
        spending_multipliers = {'low': 0.7, 'medium': 1.0, 'high': 1.4}
        base_amount *= spending_multipliers.get(profile['spending_level'], 1.0)
        
        # Weekend premium
        if timestamp.weekday() >= 5:
            base_amount *= profile.get('weekend_multiplier', 1.0)
        
        # Time-based adjustments
        if 17 <= timestamp.hour <= 20:  # Peak hours
            base_amount *= 1.1
        
        # Add some randomness
        base_amount *= random.uniform(0.8, 1.3)
        
        return max(pattern['min'], base_amount)
    
    def _get_mcc_code(self, category):
        """Get MCC (Merchant Category Code)"""
        mcc_codes = {
            'grocery': 5411,
            'retail': 5999,
            'food': 5812,
            'gas': 5541,
            'entertainment': 7929,
            'healthcare': 5912,
            'transportation': 4121,
            'utilities': 4900,
            'fitness': 7997,
            'other': 5999
        }
        return mcc_codes.get(category, 5999)
    
    def _select_channel(self, category, timestamp):
        """Select transaction channel based on category and time"""
        channel_preferences = {
            'grocery': ['in-store'],
            'retail': ['online', 'in-store', 'mobile'],
            'food': ['in-store', 'mobile'],
            'gas': ['in-store'],
            'entertainment': ['online', 'mobile', 'in-store'],
            'healthcare': ['in-store'],
            'transportation': ['mobile'],
            'utilities': ['online'],
            'fitness': ['in-store'],
            'other': ['in-store', 'online']
        }
        
        # Late hours prefer online/mobile
        if timestamp.hour >= 22 or timestamp.hour <= 6:
            online_channels = ['online', 'mobile']
            possible_channels = list(set(channel_preferences.get(category, self.channels)) & set(online_channels))
            if possible_channels:
                return random.choice(possible_channels)
        
        return random.choice(channel_preferences.get(category, self.channels))
    
    def _get_subcategory(self, category, merchant):
        """Generate subcategory based on category and merchant"""
        subcategories = {
            'grocery': ['produce', 'dairy', 'meat', 'general'],
            'retail': ['electronics', 'clothing', 'home', 'general'],
            'food': ['coffee', 'fast_food', 'restaurant', 'delivery'],
            'gas': ['fuel', 'convenience'],
            'entertainment': ['streaming', 'movies', 'gaming', 'events'],
            'healthcare': ['pharmacy', 'medical', 'dental'],
            'transportation': ['rideshare', 'public_transit', 'parking'],
            'utilities': ['electric', 'internet', 'phone', 'water'],
            'fitness': ['gym', 'classes', 'equipment'],
            'other': ['home_improvement', 'fees', 'misc']
        }
        
        return random.choice(subcategories.get(category, ['general']))
    
    def _is_recurring_transaction(self, merchant, category):
        """Determine if transaction is recurring"""
        recurring_merchants = ['Netflix', 'Spotify', 'PG&E', 'Comcast', 'Verizon', 'AT&T', '24 Hour Fitness', 'Planet Fitness']
        recurring_categories = ['utilities', 'fitness', 'entertainment']
        
        return merchant in recurring_merchants or (category in recurring_categories and random.random() < 0.3)
    
    def generate_demo_scenarios(self):
        """Generate different demo scenarios for interviews"""
        scenarios = {
            'high_spender': self.generate_realistic_transactions(500, 60),
            'budget_conscious': self._generate_budget_scenario(),
            'seasonal_shopper': self._generate_seasonal_scenario(),
            'anomaly_rich': self._generate_anomaly_scenario()
        }
        return scenarios
    
    def _generate_budget_scenario(self):
        """Generate scenario showing budget-conscious behavior"""
        df = self.generate_realistic_transactions(300, 45)
        # Reduce amounts for last 2 weeks to show budget control
        recent_mask = df['timestamp'] >= (df['timestamp'].max() - timedelta(days=14))
        df.loc[recent_mask, 'amount'] *= 0.6
        return df
    
    def _generate_seasonal_scenario(self):
        """Generate scenario with seasonal spending patterns"""
        df = self.generate_realistic_transactions(400, 75)
        # Increase retail spending in last month (holiday shopping)
        recent_mask = (df['timestamp'] >= (df['timestamp'].max() - timedelta(days=30))) & (df['category'] == 'retail')
        df.loc[recent_mask, 'amount'] *= 1.8
        return df
    
    def _generate_anomaly_scenario(self):
        """Generate scenario with clear anomalies for detection demo"""
        df = self.generate_realistic_transactions(350, 60)
        
        # Add some clear anomalies
        anomaly_indices = random.sample(range(len(df)), 5)
        for idx in anomaly_indices:
            df.loc[idx, 'amount'] *= random.uniform(5, 10)  # 5-10x normal amount
            df.loc[idx, 'merchant'] = 'Luxury Store'
            df.loc[idx, 'category'] = 'retail'
        
        return df

def create_demo_data():
    """Main function to create demo data"""
    generator = FinanceDataGenerator()
    
    # Generate main dataset
    print("Generating realistic transaction data...")
    df = generator.generate_realistic_transactions(1200, 90)
    
    # Save to CSV
    output_path = '../data/demo_transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Demo data saved to {output_path}")
    
    # Print summary statistics
    print("\n📊 Dataset Summary:")
    print(f"Total Transactions: {len(df):,}")
    print(f"Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"Total Amount: ${df['amount'].sum():,.2f}")
    print(f"Average Transaction: ${df['amount'].mean():.2f}")
    print(f"Categories: {', '.join(df['category'].unique())}")
    print(f"Merchants: {df['merchant'].nunique()} unique merchants")
    
    # Category breakdown
    print("\n🏷️ Category Breakdown:")
    category_summary = df.groupby('category').agg({
        'amount': ['sum', 'count', 'mean']
    }).round(2)
    category_summary.columns = ['Total', 'Transactions', 'Avg Amount']
    print(category_summary.sort_values('Total', ascending=False))
    
    return df

if __name__ == "__main__":
    create_demo_data()
