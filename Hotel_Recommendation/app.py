from recommend import *
import streamlit as st

# Load the collaborative model
svd_model = joblib.load('collaborative_recommendation.pkl')

# Streamlit Interface
df['hotel_name'] = df['name_x']

# Top Section
st.title("Hybrid Hotel Recommendation System")

user_code = st.selectbox("Select User", ["New User"] + df['usercode'].unique().tolist())

if st.button("Get Recommendations"):
    if user_code == "New User" or user_code not in df['usercode'].unique():
        st.warning("User not found! Check Displayed top 5 popular hotels as recommendations.")
    else:
        recommendations = hybrid_recommendations(user_code, svd_model, df, encoded_features_scaled)
    
    st.write("**Recommended Hotels:**", recommendations)

# Display top 5 hotels beside the recommendations
top_5_hotels = df['hotel_name'].value_counts().head(5).index.tolist()
st.write("**Top 5 Popular Hotels:**", top_5_hotels)

# Middle Section
st.write("---")
st.markdown("### Explore More Hotels", unsafe_allow_html=True)


st.markdown("<h5>Filter and check hotel prices by place per day</h5>", unsafe_allow_html=True)
selected_place = st.selectbox("Select Place", df['place'].unique())
selected_price = st.slider("Select Price Range", int(df['price'].min())-1, int(df['price'].max())+1)

filtered_data = df[(df['place'] == selected_place) & (df['price'] <= selected_price)]
filtered_data = filtered_data[['hotel_name', 'place', 'price']].drop_duplicates()
st.write(filtered_data)

# Bottom Section
st.write("---")
st.markdown("### Check bookings history", unsafe_allow_html=True)

# User Booking History Table
col1, spacer2, col2 = st.columns([2, 0.5, 2])

with col1:
    st.markdown("<h5>Booking History</h5>", unsafe_allow_html=True)
    if user_code != "New User" and user_code in df['usercode'].unique():
        user_data = df[df['usercode'] == user_code]
        st.write(user_data[['date', 'hotel_name', 'place', 'price', 'total', 'days']])
    else:
        st.write("No booking history available for this user.")

with col2:
    st.markdown("<h5>Spendings by Hotel</h5>", unsafe_allow_html=True)
    if user_code != "New User" and user_code in df['usercode'].unique():
        fig, ax = plt.subplots(figsize=(10, 5))  # Increase the figure size
        user_spending = user_data.groupby('hotel_name')['total'].sum()
        sns.barplot(x=user_spending.index, y=user_spending.values, ax=ax)
        ax.set_xlabel('Hotel Name')
        ax.set_ylabel('Total Spending')
        st.pyplot(fig)
    else:
        st.write("No spending data available for this user.")
        