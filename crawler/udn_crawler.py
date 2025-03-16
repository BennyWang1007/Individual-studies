"""
UDN News Scraper Module

This module provides the UDNCrawler class for fetching, parsing, and saving news articles from the UDN website.
The class extends the NewsCrawlerBase and includes functionalities to search for news articles based on a search term,
parse the details of individual articles, and save them to a database using SQLAlchemy ORM.

Classes:
    UDNCrawler: A class to scrape news from UDN.

Exceptions:
    DomainMismatchException: Raised when the URL domain does not match the expected domain for the crawler.

Usage Example:
    crawler = UDNCrawler(timeout=10)
    headlines = crawler.startup("technology")
    for headline in headlines:
        news = crawler.parse(headline.url)
        crawler.save(news, db_session)

UDNCrawler Methods:
    __init__(self, timeout: int = 5): Initializes the crawler with a default timeout for HTTP requests.
    startup(self, search_term: str) -> list[Headline]: Fetches news headlines for a given search term across multiple pages.
    get_headline(self, search_term: str, page: int | tuple[int, int]) -> list[Headline]: Fetches news headlines for specified pages.
    _fetch_news(self, page: int, search_term: str) -> list[Headline]: Helper method to fetch news headlines for a specific page.
    __create_search_params(self, page: int, search_term: str): Creates the parameters for the search request.
    __perform_request(self, params: dict): Performs the HTTP request to fetch news data.
    __parse_headlines(response): Parses the response to extract headlines.
    parse(self, url: str) -> News: Parses a news article from a given URL.
    __extract_news(soup, url: str) -> News: Extracts news details from the BeautifulSoup object.
    save(self, news: News, db: Session): Saves a news article to the database.
    _commit_changes(db: Session): Commits the changes to the database with error handling.
"""

import json
import os
import requests

from bs4 import BeautifulSoup
from enum import Enum

from .crawler_base import NewsCrawlerBase, Headline, News, NewsWithSummary
from .exceptions import HTTPException
from .utils import Logger


class UDNCategory:
    __slots__ = ["name", "url"]
    def __init__(self, name: str, url: str) -> None:
        self.name = name
        self.url = url


class UDNCategorys:
    INSTANT =   UDNCategory("instant", "https://udn.com/news/breaknews/1")
    BREAKING =  UDNCategory("breaking", "https://udn.com/api/more")
    IMPORTANT = UDNCategory("important", "https://udn.com/news/cate/2/6638")
    STARS =     UDNCategory("stars", "https://stars.udn.com/star/index")
    SPORTS =    UDNCategory("sports", "https://udn.com/news/cate/2/7227")
    WORLD =     UDNCategory("world", "https://udn.com/news/cate/2/7225")
    SOCIETY =   UDNCategory("society", "https://udn.com/news/cate/2/6639")
    LOCAL =     UDNCategory("local", "https://udn.com/news/cate/2/6641")
    FINANCE =   UDNCategory("finance", "https://udn.com/news/cate/2/6644")
    STOCK =     UDNCategory("stock", "https://udn.com/news/cate/2/6645")
    HOUSE =     UDNCategory("house", "https://house.udn.com/house/index")
    LIFE =      UDNCategory("life", "https://udn.com/news/cate/2/6649")
    PET =       UDNCategory("pet", "https://pets.udn.com/pets/index")
    HEALTH =    UDNCategory("health", "https://health.udn.com/health/index")
    EDUCATION = UDNCategory("education", "https://udn.com/news/cate/2/11195")
    CRITICISM = UDNCategory("criticism", "https://udn.com/news/cate/2/6643")
    CHINA =     UDNCategory("china", "https://udn.com/news/cate/2/6640")
    TECH =      UDNCategory("tech", "https://tech.udn.com/tech/index")
    READING =   UDNCategory("reading", "https://reading.udn.com/read/index")
    TRAVEL =    UDNCategory("travel", "https://udn.com/news/cate/1013")
    MAGAZINE =  UDNCategory("magazine", "https://udn.com/news/cate/1015")


class UDNCrawler(NewsCrawlerBase):

    _instance = None

    CHANNEL_ID = 2
    news_website_url: str = "https://udn.com/api/more"

    SAVED_NEWS_DIR = os.path.join(os.path.dirname(__file__), "saved_news")
    SAVE_NEWS_FILE = os.path.join(SAVED_NEWS_DIR, "udn_news.jsonl")
    CRAWLED_URLS_FILE = os.path.join(SAVED_NEWS_DIR, "crawled_urls.json")

    # logger: Logger = Logger(__name__)

    crawled_urls: list[tuple[str, str]] = [] # (url, filepath)
    
    logger: Logger
    skipped: bool = False
    news: News | NewsWithSummary

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(UDNCrawler, cls).__new__(cls)
        return cls._instance

    def __init__(self, timeout: int = 5) -> None:
        super(UDNCrawler, self).__init__(timeout)

    @staticmethod
    def _get_links(soup: BeautifulSoup, num: int) -> list[tuple[str, str]]:
        news_blocks = soup.find_all("div", class_="context-box__content story-list__holder story-list__holder--full")

        if len(news_blocks) != 1:
            UDNCrawler.logger.error(f"Failed to find news block in the HTML content.")
            UDNCrawler.logger.error(f"{len(news_blocks)=}")
            UDNCrawler.logger.debug(f"{soup=}")
            raise HTTPException(status_code=502, detail="Failed to parse news data from external source.")

        news_block = news_blocks[0]
        news_list = news_block.find_all("div", class_="story-list__text")

        ret: list[tuple[str, str]] = []
        count = 0

        for news in news_list:
            title = news.find("a").text
            url = news.find("a")["href"]
            url = UDNCrawler._process_url(url)
            ret.append((title, url))
            count += 1
            if count >= num: break
            UDNCrawler.logger.debug(f"Found news article: {title} from {url}")

        return ret
    
    @staticmethod
    def _process_url(url: str) -> str:
        if not url.startswith("https://udn.com"):
            url = "https://udn.com" + url
        if "?from" in url:
            url = url[:url.find("?")]
        return url

    # @staticmethod
    # def get_urls_search(keyword: str, num: int = 10) -> list[tuple[str, str]]:
    #     url = f"https://udn.com/search/word/2/{keyword}"
    #     response = requests.get(url)
    #     if response.status_code != 200: return []
    #     soup = BeautifulSoup(response.text, "html.parser")
    #     return UDNCrawler._get_links(soup, num)


    @staticmethod
    def get_headlines_catagory(mode: UDNCategory, num: int = 10, page: int = 1) -> list[Headline]:
        
        # name, url = mode.name, mode.url
        # response = UDNCrawler.__perform_request(url=url)
        # soup = BeautifulSoup(response.text, "html.parser")
        # links = UDNCrawler._get_links(soup, 20000)
        # headlines = [Headline(title=title, url=url) for title, url in links]

        headlines: list[Headline] = []

        while len(headlines) < num:
            params = UDNCrawler.__create_nextpage_param(page)
            response = UDNCrawler.__perform_request(params=params)
            hds = UDNCrawler.__parse_headlines(response)
            if len(hds) == 0: break
            for hd in hds:
                hd.url = UDNCrawler._process_url(hd.url)
            headlines.extend(hds)
            page += 1

        return headlines[:num]

    def startup(self, search_term: str) -> list[Headline]:
        """
        Initializes the application by fetching news headlines for a given search term across multiple pages.
        This method is typically called at the beginning of the program when there is no data available,
        hence it fetches headlines from the first 10 pages.

        :param search_term: The term to search for in news headlines.
        :return: A list of Headline namedtuples containing the title and URL of news articles.
        :rtype: list[Headline]
        """
        return self.get_headlines_keyword(search_term, page=(1, 10))

    @staticmethod
    def get_headlines_keyword(
        search_term: str, page: int | tuple[int, int]
    ) -> list[Headline]:

        # Calculate the range of pages to fetch news from.
        # If 'page' is a tuple, unpack it and create a range representing those pages (inclusive).
        # If 'page' is an int, create a list containing only that single page number.
        # page_range = range(*page) if isinstance(page, tuple) else [page]
        try:
            page_range: list = [i for i in range(page[0], page[1] + 1)] if isinstance(page, tuple) else [page]
            headlines = []
            for page_num in page_range:
                headlines.extend(UDNCrawler.__fetch_headlines(page_num, search_term))

            UDNCrawler.logger.info(f"Fetched {len(headlines)} headlines for search term '{search_term}'.")
            return headlines
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=502, detail="Failed to fetch news from external source.")
    
    @staticmethod
    def __perform_request(url: str | None = None, params: dict | None = None) -> requests.Response:
        url = url or UDNCrawler.news_website_url
        # print(f"{url=}, {params=}")
        UDNCrawler.logger.info(f"Performing request to URL: {url} with params: {params}")
        try:
            response = requests.get(url, params=params, timeout=UDNCrawler.timeout)
            response.raise_for_status()
            return response
        
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=502, detail="Failed to perform request to external source.")

    @staticmethod
    def __fetch_headlines(page: int, search_term: str) -> list[Headline]:
        params = UDNCrawler.__create_search_params(page, search_term)
        response = UDNCrawler.__perform_request(params=params)
        headlines = UDNCrawler.__parse_headlines(response)
        return headlines

    @staticmethod
    def __create_search_params(page: int, search_term: str) -> dict:
        return {
            "page": page,
            "id": f"search:{search_term}",
            "channelId": UDNCrawler.CHANNEL_ID,
            "type": "searchword",
        }
    
    @staticmethod
    def __create_nextpage_param(page: int) -> dict:
        return {
            "page": page,
            "id": "",
            "channelId": 1,
            "type": "breaknews",
            # "totalRecNo": 15680
        }

    @staticmethod
    def __parse_headlines(response: requests.Response) -> list[Headline]:
        """ This should only be called by getting response from udn.com/api """
        try:
            data = response.json()
            headlines = []
            for news_item in data["lists"]:
                headline = Headline(
                    title=news_item["title"],
                    url=news_item["titleLink"],
                )
                headlines.append(headline)
            return headlines
        
        except ValueError as e:
            raise HTTPException(status_code=502, detail="Failed to parse news data from external source.")

    @staticmethod
    def parse(url: str, skip_if_crawled: bool = True) -> News | NewsWithSummary | None:
        UDNCrawler.skipped = False
        if skip_if_crawled:
            is_crawled = UDNCrawler._url_crawled(url)
            if is_crawled:
                UDNCrawler.skipped = True
                return UDNCrawler.news
        response = UDNCrawler._request(url)
        soup = BeautifulSoup(response.text, "html.parser")
        news = UDNCrawler.__extract_news(soup, url)
        if news is None: return None
        UDNCrawler.logger.info(f"Parsed news article from URL: {url}")
        return news

    @staticmethod
    def __extract_news(soup: BeautifulSoup, url: str) -> News | NewsWithSummary | None:
        # print(soup)
        try:
            article_title = soup.find("h1", class_="article-content__title").text
            time = soup.find("time", class_="article-content__time").text
            content_section = soup.find("section", class_="article-content__editor")
            content = " ".join(
                paragraph.text
                for paragraph in content_section.find_all("p")
                if paragraph.text.strip() != "" and "▪" not in paragraph.text
            )

            return News(
                url=url,
                title=article_title,
                time=time,
                content=content,
            )
        except AttributeError as e:
            # raise HTTPException(status_code=502, detail="Failed to extract news data from external source.")
            UDNCrawler.logger.error(f"Failed to extract news from: {url}")
            return None

    # @staticmethod
    # def _parse_file(filepath: str) -> News | NewsWithSummary:

    #     if filepath.endswith(".json"):
    #         with open(filepath, "r", encoding="utf-8") as file:
    #             data = json.load(file)
    #             if "summary" in data:
    #                 return NewsWithSummary(**data)
    #             else:
    #                 return News(**data)

    #     if not filepath.endswith(".txt"):
    #         raise ValueError(f"Invalid file format: {filepath}")
        
    #     with open(filepath, "r", encoding="utf-8") as file:
    #         lines = file.readlines()
    #         title = lines[0].split(":")[1].strip()
    #         time = lines[1].split(":")[1].strip()
    #         url = lines[2].split(":")[1].strip()
    #         else_lines = lines[4:]

    #     for i, line in enumerate(else_lines):
    #         if line.startswith("Summary:"):
    #             content_lines = else_lines[:i]
    #             else_lines = else_lines[i:]
    #             break
    #     else:
    #         content = "".join(else_lines)
    #         return News(title=title, url=url, time=time, content=content)
        
    #     content = "".join(content_lines)

    #     for i, line in enumerate(else_lines):
    #         if line.startswith("Reason:"):
    #             summary_lines = else_lines[:i]
    #             reason_lines = else_lines[i:]
    #             break
    #     else:
    #         summary_lines = else_lines
    #         reason_lines = []

    #     summary = "".join(summary_lines)
    #     reason = "".join(reason_lines)
    #     return NewsWithSummary(title=title, url=url, time=time, content=content, summary=summary, reason=reason)


    # @staticmethod
    # def _check_if_url_crawled(url: str) -> bool:
    #     # return True if the URL has not been crawled before, and set the news attribute
    #     for u, filepath in UDNCrawler.crawled_urls:
    #         if u == url:
    #             UDNCrawler.news = UDNCrawler._parse_file(filepath)
    #             return True
    #     return False

    @staticmethod
    def fetch_and_save_news(url: str, save_folder: str = "", skip_if_crawled: bool = True) -> News | NewsWithSummary | None:
        news = UDNCrawler.parse(url, skip_if_crawled)
        if not UDNCrawler.skipped and news is not None:
            UDNCrawler.save(news, save_folder)
        return news


    # @staticmethod
    # def _commit_changes(db: Session):
    #     try:
    #         db.commit()
    #     except Exception as e:
    #         db.rollback()
    #         raise HTTPException(status_code=500, detail="Failed to commit changes to the database.")
        
