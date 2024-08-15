# DiRA: Disaster Response Assistance

A Decision Support System for better Disaster Response Management.

[![Open DiRA Website](https://disaster-response-assistance.streamlit.app/)]

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

3. Upload a CSV file for analysis.

### What files can you upload?

1. Your CSV file should contain the following two columns:
   ```
   text: This column should include the full text of each tweet.
   ```
   ```
   date: This column should specify the date when the tweet was created.
   ```

2. Sample CSV file format:
   
   ![Example Image](images/sample.png)

3. Here is a sample dataset you can upload for testing.
   ```
   Link to drive folder: https://drive.google.com/drive/folders/1bhBApSiFr7cixBO_Rn_9aIwx0KrkgGsx?usp=sharing
   ```