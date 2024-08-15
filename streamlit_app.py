import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import nltk
import string
import re
from gensim import corpora
from gensim.models import LdaModel
from stopwordsiso import stopwords
from nltk.stem import WordNetLemmatizer
import random
from wordcloud import WordCloud

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

st.title("DiRA: Disaster Response Assistance")
st.write("A Decision Support System for better Disaster Response Management.")

st.subheader("Instructions for Use:")
st.markdown("""
- Upload a CSV file for analysis.
- Your CSV file should contain the following two columns:
  - **text**: This column should include the full text of each tweet. 
  - **date**: This column should specify the date when the tweet was created.
""")

st.write("Sample CSV file format:")
st.image('images/sample.png', use_column_width=True)



location_keywords = ['Abra', 'Agusan Del Norte', 'Agusan Del Sur', 'Aklan', 'Albay',
        'Antique', 'Apayao', 'Aurora', 'Basilan', 'Bataan', 'Batanes', 'Batangas', 'Benguet', 'Biliran',
        'Bohol', 'Bukidnon', 'Bulacan', 'Cagayan', 'Camarines Norte', 'Camarines Sur', 'Camiguin', 'Capiz',
        'Catanduanes', 'Cavite', 'Cebu', 'Cotabato', 'Davao', 'Davao De Oro', 'Davao Del Norte', 'Davao Del Sur',
        'Davao Occidental', 'Davao Oriental', 'Dinagat Islands', 'Eastern Samar', 'Guimaras', 'Ifugao',
        'Ilocos Norte', 'Ilocos Sur', 'Iloilo', 'Isabela', 'Kalinga', 'La Union', 'Laguna', 'Lanao Del Norte',
        'Lanao Del Sur', 'Leyte', 'Maguindanao Del Norte', 'Maguindanao Del Sur', 'Marinduque', 'Masbate',
        'Manila', 'Misamis Occidental', 'Misamis Oriental', 'Mountain Province', 'Negros Occidental',
        'Negros Oriental', 'Northern Samar', 'Nueva Ecija', 'Nueva Vizcaya', 'Mindoro Oriental', 'Mindoro',
        'Palawan', 'Pampanga', 'Pangasinan', 'Quezon', 'Quirino', 'Rizal', 'Romblon', 'Samar', 'Sarangani',
        'Siquijor', 'Sorsogon', 'South Cotabato', 'Southern Leyte', 'Sultan Kudarat', 'Sulu', 'Surigao Del Norte',
        'Surigao Del Sur', 'Tarlac', 'Tawi-tawi', 'Zambales', 'Zamboanga Del Norte', 'Zamboanga Del Sur',
        'Zamboanga Sibugay'] 

needs_keywords = {
    'Drinking Water': ['tubig', 'water', 'inumin', 'bottled water'], 
    'Food': ['food', 'pagkain', 'ulam', 'relief goods'], 
    'Shelter': ['shelter', 'evacuation', 'bahay', 'tirahan', 'house'], 
    'Cash Assistance': ['cash', 'fund', 'pera', 'money', 'cash donation'],
    'Clothes': ['clothes', 'damit', 'clothing', 'blanket'],
    'Hygiene Kits': ['sabon', 'toothpaste', 'soap', 'shampoo', 'panglaba', 'panlaba', 'toothbrush', 'hygiene', 'hygiene kits', 'clean water']
}

disaster_keywords = {
    'Typhoon': ['typhoon', 'storm', 'rain', 'flood', 'flooding', 'ulan', 'bagyo', 'egay', 'maring', 'paeng', 'odette'],
    'Earthquake': ['earthquake', 'quake', 'magnitude', 'lindol', 'lumindol'],
    'Fire': ['fire', 'wildfire'],
    'Hurricane': ['hurricane', 'cyclone'],
    'Tornado': ['tornado', 'twister', 'funnel']
}

uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None: 
    try:
        df = pd.read_csv(uploaded_file)

        st.write("Here's the data from your CSV file:")
        st.write(df)

        if 'text' in df.columns and 'date' in df.columns:  # ensure 'date' column exists
            filtered_results = []

            for location in location_keywords:
                location_condition = df['text'].str.contains(location, case=False, na=False) 
                location_filtered_df = df[location_condition]

                if not location_filtered_df.empty:
                    for index, row in location_filtered_df.iterrows():
                        tweet_text = row['text'].lower()
                        identified_disaster_type = 'unknown'

                        # Store multiple identified needs
                        identified_needs_list = []

                        # check needs keywords
                        for need, keywords in needs_keywords.items():
                            if any(keyword in tweet_text for keyword in keywords):
                                identified_needs_list.append(need)  # Store all identified needs

                        # check disaster type
                        for disaster_type, keywords in disaster_keywords.items():
                            if any(k in tweet_text for k in keywords):
                                identified_disaster_type = disaster_type
                                break

                        # Record results for each identified need
                        for identified_needs in identified_needs_list:
                            filtered_results.append({
                                'Location': location,
                                'Disaster Type': identified_disaster_type,
                                'Needs': identified_needs,
                                'Tweet Snippets': row['text'],
                                'date': row['date']
                            })

            if filtered_results:
                results_df = pd.DataFrame(filtered_results)  
                st.subheader("Results")
                st.write(results_df)

                pivot_df = results_df.pivot_table(index='Location', columns='Needs', aggfunc='size', fill_value=0)

                # Radar Chart
                st.subheader("Radar Chart of Needs for Specific Location")
                specific_location = st.selectbox("Select Location", results_df['Location'].unique())

                if specific_location in pivot_df.index:
                    # Prepare data for radar chart using all needs
                    all_needs = list(needs_keywords.keys())
                    values = [pivot_df.loc[specific_location].get(need, 0) for need in all_needs]  # Fill missing needs with 0

                    # Close the loop for radar chart
                    values += values[:1]  # Repeat the first value to close the loop
                    num_vars = len(all_needs)
                    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
                    angles += angles[:1]  # Repeat the first angle to close the loop

                    # Plot radar chart
                    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

                    # Fill shading for all needs
                    ax.fill(angles, values, color='blue', alpha=0.25)  # Fill for shading
                    ax.plot(angles, values, color='blue', linewidth=0.5)  # Outline the area
                    ax.set_yticklabels([])  # No labels on the radial axis
                    plt.xticks(angles[:-1], all_needs)  # Use all needs for labels
                    plt.title(f"Radar Chart of Needs for {specific_location}")
                    st.pyplot(fig)
                else:
                    st.warning(f"No data found for {specific_location}.")
                
                # Stacked Bar Chart
                st.subheader("Stacked Bar Bhart of Needs by Location")
                fig, ax = plt.subplots(figsize=(12, 6))  # Increased the width of the chart
                pivot_df.plot(kind='bar', stacked=True, ax=ax)  # Increased the width of the chart
                plt.title("Needs Distribution by Location")
                plt.ylabel('Count')
                plt.xlabel('Location')

                # Adjust the rotation and align labels for better readability
                plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate the x-ticks

                max_count = pivot_df.sum(axis=1).max()  # Get the maximum total for all stacked bars
                plt.ylim(0, max_count + 10)  # Add a buffer to the maximum value

                plt.tight_layout()  # Automatically adjust layout to prevent clipping

                st.pyplot(fig)  # ensure to use the current figure
                plt.clf() 

                # statistics
                stats_df = results_df.groupby(['Disaster Type', 'Needs']).size().reset_index(name='Count')
                st.subheader("statistics of needs by disaster type")
                st.write(stats_df)

                st.subheader("Pie Chart of Distribution of Needs in Disasters")

                # Pie Chart
                for disaster in stats_df['Disaster Type'].unique():
                    data = stats_df[stats_df['Disaster Type'] == disaster]
                    fig, ax = plt.subplots()
                    ax.pie(data['Count'], labels=data['Needs'], autopct='%1.1f%%')
                    plt.title(f"Distribution of Needs in {disaster}")
                    st.pyplot(fig)

                # New statistics: count of needs by location
                #location_stats_df = results_df.groupby(['location', 'needs']).size().reset_index(name='count')
                #st.subheader("statistics of needs by location")
                #st.write(location_stats_df)

                #for location in location_stats_df['location'].unique():
                    #data = location_stats_df[location_stats_df['location'] == location]
                    #fig, ax = plt.subplots()
                    #ax.pie(data['count'], labels=data['needs'], autopct='%1.1f%%')
                    #plt.title(f"distribution of needs in {location}")
                    #st.pyplot(fig)
                
                # Creating a pivot table for stacked bar chart
                # Fill missing needs with zero

                # After filtering results and creating stats_df
                overall_needs_stats = results_df.groupby('Needs').size().reset_index(name='Count')

                # Bar chart
                st.subheader("Overall Distribution of Needs")
                fig, ax = plt.subplots()
                sns.barplot(data=overall_needs_stats, x='Needs', y='Count', ax=ax, palette='viridis')
                plt.xticks(rotation=45)
                plt.title("Distribution of Needs Across All Locations")
                st.pyplot(fig)

            else:
                st.warning("No tweets found matching the specified criteria.")
        if 'text' in df.columns and 'date' in df.columns:
            # Preprocess the text data
            tweets = df['text'].tolist()

            # Define stopwords
            english_stopwords = stopwords("en")
            tagalog_stopwords = stopwords("tl")
            all_stopwords = english_stopwords | tagalog_stopwords

            # Preprocessing function
            def preprocess_text(text):
                needs_keywords = ["food", "water", "shelter", "medicine", "medical",
                      "police", "electricity", "power", "hospital",
                      'gamot', 'pagkain', 'house', 'bahay', 'damit', 'repair',
                      'sanitation', 'tubig', 'hygiene', 'clean', 'bigas', 'rice',
                      'clothing', 'clothes', 'tubig', 'gutom', 'relief', 'tulong',
                      'emergency', 'help', 'evacuation', 'need', 'donation', 
                      "communication", "rescue", "safety", "security", "protection", 
                      "law enforcement", "relief", "aid", "assistance", "support", 
                      "donation", "volunteer", "damage", "destruction", "repair", 
                      "reconstruction", "rehabilitation", "development", "information", 
                      "awareness", "preparedness", "response", "recovery"]
                if not any(keyword in text.lower() for keyword in needs_keywords):
                    return None
                text = text.lower()
                text = re.sub(r'http\s+', '', text)
                text = re.sub(r'#\w+', '', text)
                text = re.sub(r'@\w+', '', text)
                text = re.sub(r'\d+', '', text)
                text = text.encode('ascii', 'ignore').decode('ascii')
                text = text.translate(str.maketrans('', '', string.punctuation))

                tokens = nltk.word_tokenize(text)
                tokens = [word for word in tokens if word not in all_stopwords]
                lemmatizer = WordNetLemmatizer()
                tokens = [lemmatizer.lemmatize(word) for word in tokens if len(word) > 2]

                return ' '.join(tokens)

            # Filter tweets
            needs_filtered_tweets = [preprocess_text(tweet) for tweet in tweets]
            needs_filtered_tweets = [tweet for tweet in needs_filtered_tweets if tweet]

            # Create dictionary and corpus
            def create_dictionary(texts):
                tokenized_texts = [text.split() for text in texts if text is not None]
                dictionary = corpora.Dictionary(tokenized_texts)
                stop_ids = [dictionary.token2id[stopword] for stopword in tagalog_stopwords if stopword in dictionary.token2id]
                dictionary.filter_tokens(stop_ids)
                corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
                return dictionary, corpus

            dictionary, corpus = create_dictionary(needs_filtered_tweets)

            # Train the LDA model
            num_topics = 8
            lda_model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics, passes=10, iterations=50, alpha=0.8, eta=0.8)

            topic_words = []
            # num_topics should be set to the number of topics the LDA model was trained on
            num_topics_trained = lda_model.num_topics
            for i in range(num_topics_trained):
                topic_words.append(lda_model.show_topic(i, topn=10))

            st.subheader('Word Cloud for Tweet Topics')

            # Create and display word clouds
            for i, topic in enumerate(topic_words):
                word_dict = {word: abs(weight) for word, weight in topic}
                cloud = WordCloud(background_color='white').generate_from_frequencies(word_dict)  
                plt.figure(figsize=(8, 5))  # Create a new figure for each topic
                plt.title(f"Topic {i + 1}")  
                plt.imshow(cloud, interpolation='bilinear') 
                plt.axis("off")  # Hide axes

                st.pyplot(plt) 
                plt.clf() 

            # Summarize topics
            def summarize_topic(topic_words):
                needs_categories = {
                    'drinking water': ['tubig', 'water', 'inumin', 'bottled water', 'purified water'],
                    'food': ['food', 'pagkain', 'ulam', 'relief goods', 'rice', 'canned goods', 'bigas'],
                    'shelter': ['shelter', 'evacuation', 'bahay', 'tirahan', 'house'],
                    'cash assistance': ['cash', 'fund', 'pera', 'money', 'cash donation'],
                    'clothes': ['clothes', 'damit', 'clothing', 'blanket'],
                    'hygiene kits': ['sabon', 'toothpaste', 'soap', 'shampoo', 'panglaba', 'panlaba', 'toothbrush', 'hygiene', 'hygiene kits', 'clean water']
                }

                templates = {
                    'drinking water': ["There is a critical shortage of clean drinking water.",
                                      "Access to safe drinking water is limited.",
                                      "Residents are in urgent need of potable water.",
                                      "Bottled water and water purification supplies are essential."],
                    'food': [
                        "Food supplies are insufficient to meet the needs of the affected population.",
                        "There is a pressing need for food assistance, including {} and {}.",  # Placeholder for specific food items
                        "Malnutrition is a growing concern due to lack of adequate food.",
                        "Food distribution systems have been disrupted."
                    ],
                    'shelter': [
                        "There is a critical shortage of shelter for displaced individuals.",
                        "Temporary housing solutions are urgently needed.",
                        "Many homes have been damaged or destroyed.",
                        "Evacuation centers are overcrowded."
                    ],
                    'cash assistance': [
                        "Financial assistance is crucial for recovery efforts.",
                        "Cash aid can help people meet their immediate needs.",
                        "Economic support is essential for rebuilding livelihoods.",
                        "There is a need for cash assistance programs."
                    ],
                    'clothes': [
                        "Clothing and blankets are urgently needed for affected populations.",
                        "Many people have lost their clothing due to the disaster.",
                        "There is a shortage of essential clothing items.",
                        "Warm clothing is required for those in cold weather conditions."
                    ],
                    'hygiene kits': [
                        "Proper hygiene is essential to prevent the spread of diseases.",
                        "There is a critical need for hygiene kits and supplies.",
                        "Sanitation facilities are inadequate and need improvement.",
                        "Personal hygiene products are essential for affected populations."
                    ],
                }

                general_template = {
                    'general': [
                        "This topic addresses a wide range of needs related to disaster recovery.",
                        "The affected population requires comprehensive assistance.",
                        "There is a need for coordinated response efforts.",
                        "Long-term recovery and rehabilitation are essential."
                    ]
                }

                identified_needs = []
                for category, keywords in needs_categories.items():
                    if any(keyword in topic_words for keyword in keywords):
                        identified_needs.append(category)

                if identified_needs:
                    summary_parts = []
                    for need in identified_needs:
                        relevant_words = [word for word in topic_words if word in needs_categories[need]]
                        template = random.choice(templates[need])
                        if len(relevant_words) >= 2:
                            summary_part = template.format(*relevant_words[:2])
                        elif relevant_words:
                            summary_part = template.format(relevant_words[0]) if '{}' in template else template
                        else:
                            summary_part = template
                        summary_parts.append(summary_part)
                    summary = " ".join(summary_parts)
                else:
                    summary = random.choice(general_template['general'])

                return summary

            st.subheader("Tweet Topic Summaries:")
            for k in range(num_topics):
                topic_words = [word for word, _ in lda_model.show_topic(k, topn=10)]
                summary = summarize_topic(topic_words)
                st.write(f'Topic #{k + 1} Summary: {summary}')
        else:
            st.error("The uploaded CSV file must contain both 'text' and 'date' columns.")
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
else:
    st.info("Please upload a CSV file.")

