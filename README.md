### 🎬 Content-Based Movie Recommendation System ###

This project is a machine learning-based recommendation system built with Python. It suggests movies similar to a user-selected movie by analyzing content such as plot overview, genres, keywords, cast, and crew.

# Features
* **Data Processing:** Merges and cleans TMDB 5000 Movies & Credits datasets.
* **Feature Engineering:** Creates a unified `tags` column from overview, genre, and cast information.
* **Vectorization:** Uses `CountVectorizer` to convert text data into numerical vectors.
* **Similarity Calculation:** Calculates cosine similarity distances between movie vectors to find the closest matches.

# Technologies Used
* **Python 3.x**
* **Pandas** (Data Manipulation)
* **Scikit-Learn** (Machine Learning & Vectorization)
* **Numpy**

# Dataset
The project uses the TMDB 5000 Movie Dataset, which includes:
* `tmdb_5000_movies.csv`: Movie details (budget, genres, homepage, id, keywords, original_language, etc.)
* `tmdb_5000_credits.csv`: Cast and crew information.

*(Note: Data source is Kaggle)*

# How It Works
1.  **Data Loading:** The script reads the CSV files.
2.  **Preprocessing:** It extracts relevant lists from JSON-like columns (genres, keywords, cast, crew) and merges them into a single string.
3.  **Vectorization:** The text data is converted into 5000-dimensional vectors using the "Bag of Words" technique.
4.  **Recommendation:** When a movie title is entered, the system finds the 5 nearest vectors (movies) based on cosine similarity.

# How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/REPO_NAME.git](https://github.com/YOUR_USERNAME/REPO_NAME.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the script:
    ```bash
    python movie_recommender.py
    ```

# Example Output
```text
If you liked the 'Avatar', you'll like these too:

1. Titan A.E.
2. Small Soldiers
3. Independence Day
4. Ender's Game
5. Aliens vs Predator: Requiem