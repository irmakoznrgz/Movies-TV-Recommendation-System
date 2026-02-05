#   Content Recommender System (Movies & TV Shows)

A machine learning-based recommendation system capable of suggesting both **Movies** and **TV Shows**. It uses Content-Based Filtering with Natural Language Processing (NLP) techniques to find similarities in plot, genres, cast, and crew.

##  Key Features (v1.0)
* **Hybrid Dataset:** Merges TMDB Movies (5,000+) and TMDB TV Shows (Select data from 160,000+) into a unified recommendation engine.
* **Performance Optimization:** Implements **Sparse Matrix** operations and **On-the-fly Cosine Similarity** calculation. This allows the system to handle 160,000+ items without consuming excessive RAM (Memory efficient).
* **Smart Text Processing:** utilizes `CountVectorizer` to create tags from overview, genre, cast, and crew metadata.

##  Technologies
* **Python 3.x**
* **Pandas** (Data Engineering & Merging)
* **Scikit-Learn** (Vectorization & Cosine Similarity)
* **Numpy** (High-performance matrix calculations)

##  How It Works
1.  **Data Ingestion:** Loads and creates a uniform structure for both Movies and TV Shows.
2.  **Preprocessing:** Cleans text data, handles missing values, and merges relevant columns into a single `tags` vector.
3.  **Vectorization:** Converts text to numerical vectors using the "Bag of Words" technique (limited to top 5000 features for efficiency).
4.  **Recommendation:** When a title is queried, the system calculates the cosine distance between that specific title and the entire dataset in real-time, returning the top 5 matches instantly.

##  Installation & Usage
1.  Clone the repo:
    ```bash
    git clone [https://github.com/irmakoznrgz/Movies-TV-Recommendation-System.git](https://github.com/irmakoznrgz/Movies-TV-Recommendation-System.git)
    ```
2.  Install requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the recommender:
    ```bash
    python movie_recommender.py
    ```

##  Example Output
```text
If you liked 'Breaking Bad', you might like:
1. Better Call Saul (TV Show)
2. El Camino: A Breaking Bad Movie (Movie)
3. Ozark (TV Show)
4. Narcos (TV Show)
5. The Wire (TV Show)

##  Author
**[Fadime Irmak Öznergiz]** - Statistics Student - Ankara University
