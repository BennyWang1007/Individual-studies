import abc
import json
import os

from pydantic import AnyHttpUrl
from tldextract import tldextract
# from sqlalchemy.orm import Session
from .exceptions import DomainMismatchException
from pydantic import BaseModel, Field, AnyHttpUrl
# from src.error_handler.logger import Logger

from .utils import Logger

class Headline(BaseModel):
    title: str = Field(
        default=...,
        example="Title of the article",
        description="The title of the article"
    )
    url: AnyHttpUrl | str = Field(
        default=...,
        example="https://www.example.com",
        description="The URL of the article"
    )


class News(Headline):
    time: str = Field(
        default=...,
        example="2021-10-01T00:00:00",
        description="The time the article was published"
    )
    content: str = Field(
        default=...,
        example="Content of the article",
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
        example="Summary of the article",
        description="The summary of the article"
    )
    reason: str = Field(
        default=...,
        example="Reason of the article",
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

    news_website_url: AnyHttpUrl | str
    news_website_news_child_urls: list[AnyHttpUrl | str]

    SAVED_NEWS_DIR = os.path.join(os.path.dirname(__file__), "saved_news")
    CRAWLED_URLS_FILE = os.path.join(SAVED_NEWS_DIR, "crawled_urls.json")
    crawled_urls: list[tuple[str, str]]

    timeout: int
    logger: Logger
    news: News | NewsWithSummary

    @staticmethod
    def init(timeout: int = 10) -> None:
        NewsCrawlerBase.logger = Logger(__name__).get_logger()
        NewsCrawlerBase.logger.info(f"Initializing NewsCrawlerBase with timeout: {timeout} seconds.")
        NewsCrawlerBase.timeout = timeout
        if os.path.exists(NewsCrawlerBase.CRAWLED_URLS_FILE):
            with open(NewsCrawlerBase.CRAWLED_URLS_FILE, "r") as f:
                NewsCrawlerBase.crawled_urls = json.load(f)
                NewsCrawlerBase.logger.info(f"Loaded {len(NewsCrawlerBase.crawled_urls)} crawled URLs")
                # NewsCrawlerBase.logger.debug(f"Loaded crawled URLs from file: {NewsCrawlerBase.CRAWLED_URLS_FILE}")
                # NewsCrawlerBase.logger.debug(f"Crawled URLs: {NewsCrawlerBase.crawled_urls}")
        else:
            NewsCrawlerBase.crawled_urls = []

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
    def parse(self, url: AnyHttpUrl | str, skip_if_crawed: bool = True) -> News | NewsWithSummary | None:
        """
        Given a news URL from the news website, fetch and parse the detailed news content.

        This method takes a URL that belongs to a news article on the news_website_url, retrieves the full content of
        the news article, and returns it in the form of a News namedtuple. The News namedtuple includes the title,
        URL, publication time, and content of the news article.

        :param url: The URL of the news article to be fetched and parsed.
        :return: A News namedtuple containing the title, URL, time, and content of the news article.
        """

        return NotImplemented
        

    def validate_and_parse(self, url: AnyHttpUrl | str) -> News | NewsWithSummary | None:
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


    # @staticmethod
    # @abc.abstractmethod
    # def save(news: News, db: Session | None):
    #     """
    #     Save the news content to a persistent storage.

    #     This method takes a News namedtuple containing the title, URL, publication time, and content of a news article,
    #     and saves it to a persistent storage, such as a database. The method should handle the storage of the news
    #     content, ensuring that duplicate news articles are not saved.

    #     :param news: A News namedtuple containing the title, URL, time, and content of the news article.
    #     :param db: An instance of the database session to use for saving the news content.
    #     """
    #     return NotImplemented

    @staticmethod
    def illigalize_filename(filename: str) -> str:
        return filename.replace("/", "_").replace("\\", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_")

    @staticmethod
    def save(news: News | NewsWithSummary, save_folder: str = "") -> None:
        filename = f"{news.title}.txt"
        filename = NewsCrawlerBase.illigalize_filename(filename)
        save_folder = os.path.join(NewsCrawlerBase.SAVED_NEWS_DIR, save_folder)
        filepath = os.path.join(save_folder, filename)

        if os.path.exists(filepath):
            NewsCrawlerBase.logger.info(f"News article already exists: {filename}")
            NewsCrawlerBase._add_crawled_url(news.url, filepath)
            return

        if not os.path.exists(NewsCrawlerBase.SAVED_NEWS_DIR):
            os.makedirs(NewsCrawlerBase.SAVED_NEWS_DIR)

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(str(news))

        NewsCrawlerBase.logger.info(f"Saved news article to file: {filename}")
        NewsCrawlerBase._add_crawled_url(news.url, filepath)


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


    @staticmethod
    def _add_crawled_url(url: str, filepath: str):
        NewsCrawlerBase.crawled_urls.append((url, filepath))
        with open(NewsCrawlerBase.CRAWLED_URLS_FILE, "w") as f:
            json.dump(NewsCrawlerBase.crawled_urls, f)
            NewsCrawlerBase.logger.info(f"Added crawled URL to file: {url}")

    
    @staticmethod
    def _url_crawled(url: str) -> bool:
        # return True if the URL has not been crawled before, and set the news attribute
        for u, filepath in NewsCrawlerBase.crawled_urls:
            if u == url:
                NewsCrawlerBase.news = NewsCrawlerBase._parse_file(filepath)
                return True
        return False


    def _is_valid_url(self, url: AnyHttpUrl | str) -> bool:
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