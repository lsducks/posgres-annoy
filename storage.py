from typing import Sequence, List, Literal
from functools import lru_cache

import psycopg2
import pandas as pd
import pymorphy3
import jinja2
import joblib

from annoy import AnnoyIndex
from razdel import tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords


class Storage:
    def __init__(
        self,
        table_names: list,
        host: str = 'localhost',
        port: int = 5432,
        dbname: str = 'postgres',
        username: str = 'postgres',
        password: str = 'postgres'
    ) -> None:
        connection = psycopg2.connect(
            host=host, port=port, user=username, dbname=dbname, password=password
        )

        self.cursor = connection.cursor()

        # self.columns = (
        #     'title_id titletype primarytitle originaltitle ' +
        #     'isadult startyear endyear runtimeminutes genres'
        # ).split()

        self.columns = self.__get_columns(table_names)

        self.morph = pymorphy3.MorphAnalyzer()
        self.stopwords = set(stopwords.words("russian"))


    def __get_columns(
        self,
        table_names: list,
    ) -> list:
        query_template = jinja2.Template("""
           SELECT column_name
           FROM INFORMATION_SCHEMA.COLUMNS
           WHERE TABLE_NAME = '{{ table_names[0] }}'
           {% if table_names|length > 1 -%}
           {% for table_name in table_names[1:] -%}
           UNION
           SELECT column_name
           FROM INFORMATION_SCHEMA.COLUMNS
           WHERE TABLE_NAME = '{{ table_name }}' 
           {%- endfor %}
           {%- endif %}
        """)

        query = query_template.render(table_names=table_names)

        self.cursor.execute(query)

        return self.cursor.fetchall()

    def query(
        self, 
        min_year: int | None = None,
        max_year: int | None = None,
        is_adult: Literal[0, 1] | None = None,
        min_time: int | None = None,
        max_time: int | None = None,
        title_type: str | None = None,
        min_rating: float | None = None,
        n_votes: int | None = None,
        limit: int | None = None,
        genre: str | None = None
    ) -> pd.DataFrame:
        query_template = jinja2.Template("""
        select *
            from titles
            {% if min_rating -%}
            inner join ratings using(title_id)
            {%- endif %}
            where endyear IS NULL
            {% if is_adult -%}
            and isadult = {{ is_adult }}
            {%- endif %}
            {% if title_type -%}
            and titletype = '{{ title_type }}'
            {%- endif %}
            {% if min_time -%}
            and runtimeminutes >= {{ min_time }}
            {%- endif %}
            {% if max_time -%}
            and runtimeminutes <= {{ max_time }}
            {%- endif %}
            {% if min_year -%}
            and startyear >= {{ min_year }}
            {%- endif %}
            {% if max_year -%}
            and startyear <= {{ max_year }}
            {%- endif %}
            {% if min_rating -%}
            and avg_rating >= {{ min_rating }}
            {%- endif %}
            {% if n_votes -%}
            and n_votes >= {{ n_votes }}
            {%- endif %}
            order by startyear desc
            {% if limit %}
            limit {{ limit }}
            {% endif %}
        """)
        query = query_template.render(
            is_adult=is_adult,
            title_type=title_type,
            min_time=min_time,
            max_time=max_time,
            min_year=min_year,
            max_year=max_year,
            min_rating=min_rating,
            n_votes=n_votes,
            limit=limit
        )
        
        self.cursor.execute(query)
        result = pd.DataFrame(
            data=self.cursor.fetchall(),
            columns=self.columns
        )

        if not genre:
            return result
        
        title_ids = self.text_predict(genre)
        result = result[result['title_id'].isin(title_ids)]
        return result

    def text_predict(self, text: str, k: int = 50000) -> List[str]:
        text = self.preprocess_text(text)
        tf_idf_vec = self.tf_idf_model.transform([text])
        lsa_vec = self.lsa_model.transform(tf_idf_vec)
        result_ids = self.annoy_storage.get_nns_by_vector(lsa_vec[0], n=k)
        result_pnlots = [self.id2title[ids] for ids in result_ids]
        return result_pnlots

    def fit_text_embeddings(self, lsa_dim: int = 25, query_limit: int = 4000000) -> None:
        df = self.query(limit=query_limit)
        title_ids = df['title_id'].values
        self.title2id = {title_id: i for i, title_id in enumerate(title_ids)}
        self.id2title = {i: title_id for i, title_id in enumerate(title_ids)}

        texts = df['genres'].values
        del df

        texts = self.preprocess_texts(texts)
        
        self.tf_idf_model = TfidfVectorizer()
        tf_idf_vectors = self.tf_idf_model.fit_transform(texts)

        self.lsa_model = TruncatedSVD(n_components=lsa_dim, n_iter=10)
        lsa_vectors = self.lsa_model.fit_transform(tf_idf_vectors)

        self.annoy_storage = AnnoyIndex(lsa_dim, 'angular')
        for title_id, lsa_vec in zip(title_ids, lsa_vectors):
            self.annoy_storage.add_item(self.title2id[title_id], lsa_vec)

        self.annoy_storage.build(30)

    def preprocess_texts(self, texts: Sequence[str]) -> List[str]:
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        return preprocessed_texts
    
    def preprocess_text(self, text: str) -> str:
        text = text.lower().strip()
        tokens = [t.text for t in tokenize(text)]
        tokens = [t for t in tokens if t not in self.stopwords]
        lemmatized_tokens = [self.get_normal_form(t) for t in tokens]
        preprocessed_text = ' '.join(lemmatized_tokens)
        return preprocessed_text

    @lru_cache(300_000)
    def get_normal_form(self, word: str) -> str:
        return self.morph.parse(word)[0].normal_form

    def save(self, filepath_directory: str) -> None:
        saved_attrs = {}

        saved_attrs['tf_idf_model'] = self.tf_idf_model
        saved_attrs['lsa_model'] = self.lsa_model
        saved_attrs['title2id'] = self.title2id
        saved_attrs['id2title'] = self.id2title

        joblib.dump(saved_attrs, filepath_directory + 'attrs.joblib')
        self.annoy_storage.save(filepath_directory + 'annoy.ann')

    def load(self, filepath_attrs: str, filepath_annoy, lsa_dim: int = 25) -> None:
        saved_attrs = joblib.load(filepath_attrs)

        for attribute, value in saved_attrs.items():
            setattr(self, attribute, value)

        self.annoy_storage = AnnoyIndex(lsa_dim, 'angular')
        self.annoy_storage.load(filepath_annoy)


example = Storage(['titles', 'ratings'])

print(example.columns)

example.load('./data/attrs.joblib', './data/annoy.ann')

print(example.query(title_type='movie', min_year=2000, is_adult=0, min_time=90,
                    max_time=180, min_rating=4, n_votes=1000, limit=50))
