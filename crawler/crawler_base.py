import abc
import json
import os

from pydantic import AnyHttpUrl
from tldextract import tldextract
# from sqlalchemy.orm import Session
from .exceptions import DomainMismatchException, HTTPException
from pydantic import BaseModel, Field, AnyHttpUrl
# from src.error_handler.logger import Logger

from .utils import Logger

class Headline(BaseModel):
    title: str = Field(
        default=...,
        examples=["Title of the article"],
        description="The title of the article"
    )
    url: str = Field(
        default=...,
        examples=["https://www.example.com"],
        description="The URL of the article"
    )


class News(Headline):
    time: str = Field(
        default=...,
        examples=["2021-10-01T00:00:00"],
        description="The time the article was published"
    )
    content: str = Field(
        default=...,
        examples=["Content of the article"],
        description="The content of the article"
    )

    def __str__(self) -> str:
        return \
f"""\
Title: {self.title}
Time: {self.time}
URL: {self.url}

Content: {self.content}

"""


class NewsWithSummary(News):
    summary: str = Field(
        default=...,
        examples=["Summary of the article"],
        description="The summary of the article"
    )
    reason: str = Field(
        default=...,
        examples=["Reason of the article"],
        description="The reason of the article"
    )

    def __str__(self) -> str:
        return super().__str__() + \
f"""\
Summary: {self.summary}

Reason: {self.reason}\
"""



import requests
from bs4 import BeautifulSoup

class NewsCrawlerBase(metaclass=abc.ABCMeta):

    news_website_url: str
    news_website_news_child_urls: list[str]

    SAVED_NEWS_DIR = os.path.join(os.path.dirname(__file__), "saved_news")
    SAVE_NEWS_FILE = "default_news.jsonl"   # default file to save news
    
    CRAWLED_URLS_FILE = os.path.join(SAVED_NEWS_DIR, "crawled_urls.json")
    CRAWLED_URLS_ONLY_FILE = os.path.join(SAVED_NEWS_DIR, "crawled_urls_only.json")

    crawled_urls: list[tuple[str, str]]
    crawled_urls_only: list[str]

    timeout: int
    logger: Logger
    news: News | NewsWithSummary

    
    @classmethod
    def class_init(cls, timeout: int = 10):
        cls.logger = Logger(__name__)
        cls.logger.info(f"Initializing {cls} class.")

        cls.timeout = timeout
        cls.logger.info(f"Setting timeout to {timeout} seconds.")

        if os.path.exists(cls.CRAWLED_URLS_FILE):
            with open(cls.CRAWLED_URLS_FILE, "r", encoding="utf-8") as f:
                cls.crawled_urls = json.load(f)
                cls.logger.info(f"Loaded {len(cls.crawled_urls)} crawled URLs")
        else:
            cls.crawled_urls = []

        if os.path.exists(cls.CRAWLED_URLS_ONLY_FILE):
            with open(cls.CRAWLED_URLS_ONLY_FILE, "r", encoding="utf-8") as f:
                cls.crawled_urls_only = json.load(f)
                cls.logger.info(f"Loaded {len(cls.crawled_urls_only)} crawled URLs only")
        else:
            cls.crawled_urls_only = []

    def __init__(self, timeout: int = 10) -> None:
        self.class_init(timeout)
        # self.logger = Logger(__name__).get_logger()
        # self.logger.info(f"Initializing NewsCrawlerBase with timeout: {timeout} seconds.")
        # NewsCrawlerBase.timeout = timeout
        # if os.path.exists(self.CRAWLED_URLS_FILE):
        #     with open(self.CRAWLED_URLS_FILE, "r", encoding="utf-8") as f:
        #         self.crawled_urls = json.load(f)
        #         self.logger.info(f"Loaded {len(self.crawled_urls)} crawled URLs")
        #         # NewsCrawlerBase.logger.debug(f"Loaded crawled URLs from file: {NewsCrawlerBase.CRAWLED_URLS_FILE}")
        #         # NewsCrawlerBase.logger.debug(f"Crawled URLs: {NewsCrawlerBase.crawled_urls}")
        # else:
        #     self.crawled_urls = []

        # if os.path.exists(self.CRAWLED_URLS_ONLY_FILE):
        #     with open(self.CRAWLED_URLS_ONLY_FILE, "r", encoding="utf-8") as f:
        #         self.crawled_urls_only = json.load(f)
        #         self.logger.info(f"Loaded {len(self.crawled_urls_only)} crawled URLs")
        # else:
        #     self.crawled_urls_only = []

    @abc.abstractmethod
    def get_headlines_keyword(
            self, search_term: str, page: int | tuple[int, int]
    ) -> list[Headline]:
        """
        Searches for news headlines on the news website based on a given search term and returns a list of headlines.

        This method searches through the entire news_website_url using the specified search term, and returns a list
        of Headline namedtuples, where each Headline includes the title and URL of a news article. The page parameter
        can be an integer representing a single page number or a tuple representing a range of page numbers to search
        through.
        # The offset and limit parameters apply to the resulting list of headlines, allowing you to skip a
        # certain number of headlines and limit the number of headlines returned, respectively.

        :param search_term: A search term to search for news articles.
        :param page: A page number (int) or a tuple of start and end page numbers (tuple[int, int]).
        # :param offset: The number of headlines to skip from the beginning of the list.
        # :param limit: The maximum number of headlines to return.
        :return: A list of Headline namedtuple  s, each containing a title and a URL.
        """
        return NotImplemented

    @abc.abstractmethod
    def parse(self, url: str, skip_if_crawed: bool = True) -> News | NewsWithSummary | None:
        """
        Given a news URL from the news website, fetch and parse the detailed news content.

        This method takes a URL that belongs to a news article on the news_website_url, retrieves the full content of
        the news article, and returns it in the form of a News namedtuple. The News namedtuple includes the title,
        URL, publication time, and content of the news article.

        :param url: The URL of the news article to be fetched and parsed.
        :return: A News namedtuple containing the title, URL, time, and content of the news article.
        """

        return NotImplemented
        

    def validate_and_parse(self, url: str) -> News | NewsWithSummary | None:
        """
        Validates the given URL and ensures that it belongs to the news website or its child URLs. If the URL is valid,
        it proceeds with parsing the news content by invoking the `parse` method from the child class.

        This method first checks if the provided URL is valid for the current news website. If the URL is not valid,
        it raises a `DomainMismatchException`. If the URL is valid, the method passes the URL to the `parse` method
        (which should be implemented in the child class) to retrieve and parse the news content.

        :param url: The URL of the news article to be validated and parsed.
        :return: A `News` object containing the parsed news details (title, URL, time, and content).
        :raises DomainMismatchException: If the URL does not belong to the allowed domain or its child URLs.
        """
        # logger = Logger(__name__, "validate_and_parse").get_logger()
        if not self._is_valid_url(url):
            raise DomainMismatchException(url)
        self.logger.info(f"Valid URL: {url}")
        return self.parse(url)
    
    @classmethod
    def _request(cls, url: str, params: dict | None = None) -> requests.Response:
        try:
            response = requests.get(url, params=params, timeout=cls.timeout)
            response.raise_for_status()
            return response
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=502, detail="Failed to perform request to external source.")


    @staticmethod
    def legalize_filename(filename: str) -> str:
        return filename.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")


    @classmethod
    def save(cls, news: News | NewsWithSummary, save_folder: str = "", separate_file: bool = False) -> None:

        if separate_file:
            filename = f"{news.title}.txt"
            filename = cls.legalize_filename(filename)
            filepath = os.path.join(save_folder, filename)
        else:
            filename = cls.SAVE_NEWS_FILE
            filepath = os.path.join(cls.SAVED_NEWS_DIR, filename)
        
        save_folder = os.path.join(cls.SAVED_NEWS_DIR, save_folder)

        # if os.path.exists(filepath):
        #     cls.logger.info(f"News article already exists: {filename}")
        #     cls._add_crawled_url(news.url, filepath)
        #     return

        if not os.path.exists(cls.SAVED_NEWS_DIR):
            os.makedirs(cls.SAVED_NEWS_DIR)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if separate_file:
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(str(news))
        else:
            # append the news to the jsonl file
            with open(filepath, "a", encoding="utf-8") as file:
                file.write(json.dumps(news.dict(), ensure_ascii=False))
                file.write("\n")

        cls.logger.info(f"Saved news article to file: {filename}")
        cls._add_crawled_url(news.url, filepath)


    @staticmethod
    def _parse_file(filepath: str) -> News | NewsWithSummary:
        # handle json file
        if filepath.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
                if "summary" in data:
                    return NewsWithSummary(**data)
                else:
                    return News(**data)
        
        # only accept json or txt file
        if not filepath.endswith(".txt"):
            raise ValueError(f"Invalid file format: {filepath}")
        
        # handle txt file
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()
            title = lines[0].split(":")[1].strip()
            time = lines[1].split(":")[1].strip()
            url = lines[2].split(":")[1].strip()
            else_lines = lines[4:]

        for i, line in enumerate(else_lines):
            if line.startswith("Summary:"):
                content_lines = else_lines[:i]
                else_lines = else_lines[i:]
                break
        else:
            content = "".join(else_lines)
            return News(title=title, url=url, time=time, content=content)
        
        content = "".join(content_lines)

        for i, line in enumerate(else_lines):
            if line.startswith("Reason:"):
                summary_lines = else_lines[:i]
                reason_lines = else_lines[i:]
                break
        else:
            summary_lines = else_lines
            reason_lines = []

        summary = "".join(summary_lines)
        reason = "".join(reason_lines)
        return NewsWithSummary(title=title, url=url, time=time, content=content, summary=summary, reason=reason)


    @classmethod
    def _add_crawled_url(cls, url: str, filepath: str):
        cls.crawled_urls.append((url, filepath))
        cls.crawled_urls_only.append(url)

        with open(cls.CRAWLED_URLS_FILE, "w") as f:
            json.dump(cls.crawled_urls, f, ensure_ascii=False, indent=4)

        with open(cls.CRAWLED_URLS_ONLY_FILE, "w") as f:
            json.dump(cls.crawled_urls_only, f, ensure_ascii=False, indent=4)
    
        # cls.logger.info(f"Added crawled URL to file: {url}")

    
    @classmethod
    def _url_crawled(cls, url: str) -> bool:
        """ Return True if the URL has not been crawled before, and set the news attribute """
        if url in cls.crawled_urls_only:
            with open(cls.SAVE_NEWS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    if data["url"] == url:
                        if "summary" in data:
                            cls.news = NewsWithSummary(**data)
                        else:
                            cls.news = News(**data)
                        # cls.logger.info(f"News found in class{cls}")
                        return True
            raise RuntimeError(f"URL has been crawled before but not found in class{cls}")
        return False


    def _is_valid_url(self, url: str) -> bool:
        """
        Check if the given URL belongs to the news website or its child URLs.

        This method checks if the given URL belongs to the news_website_url or any of its child URLs. It returns True if
        the URL is valid, and False otherwise.

        :param url: The URL to be checked for validity.
        :return: True if the URL is valid, False otherwise.
        """
        main_domain = tldextract.extract(self.news_website_url).registered_domain
        url_domain = tldextract.extract(url).registered_domain

        if url_domain == main_domain:
            return True
        return False