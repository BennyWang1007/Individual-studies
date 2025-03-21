import random
import time

from .crawler_base import News, NewsWithSummary
from .udn_crawler import UDNCrawler, UDNCategory, UDNCategorys
from .utils import Logger

def test_get_news_with_category(catagory: UDNCategory, num: int = 10, logger: Logger = Logger("__main__")) -> None:
    logger.info(f"Start to get {num} news from {catagory.name}")
    count = 0
    page = 1
    while count < num:
        headlines = UDNCrawler.get_headlines_catagory(catagory, num - count, page=page)
        for headline in headlines:
            news = UDNCrawler.fetch_and_save_news(headline.url, catagory.name, skip_if_crawled=True)
            if UDNCrawler.skipped:
                logger.info(f"Skipped {headline.title}, {headline.url}:")
                continue
            count += 1
            rand_sleep = 0.5 + random.random() * 0.5
            time.sleep(rand_sleep)
        page += 1
        rand_sleep = 0.5 + random.random() * 0.5
        time.sleep(rand_sleep)
    logger.info(f"Get {count} news from {catagory.name}")

def test_get_news_with_keywords(keywords: str, num: int = 10) -> None:
    page = 1
    headlines = UDNCrawler.get_headlines_keyword(keywords, page=page)
    for headline in headlines:
        news = UDNCrawler.fetch_and_save_news(headline.url, keywords)
        if UDNCrawler.skipped:
            print(f"Skipped {headline.title}, {headline.url}:")
            continue
        rand_sleep = 1.0 + random.random() * 1.0
        time.sleep(rand_sleep)

def main() -> None:
    UDNCrawler()
    test_get_news_with_category(UDNCategorys.INSTANT, 100)

    # test_get_news_with_keywords("NBA", 10)
    # urls_instant: list[tuple] = UDNCrawler.get_headlines(UDNCategorys.INSTANT, 10)
    # # keywords = "NBA"
    # for name, url in urls_instant:
    #     # print(name, url)
    #     news = UDNCrawler.fetch_and_save_news(url, UDNCategorys.INSTANT.name)
    #     if UDNCrawler.skipped:
    #         print(f"Skipped {name}, {url}:")
    #         continue
    #     rand_sleep = 1.0 + random.random() * 3.0
    #     time.sleep(rand_sleep)

    # keywords = "NBA"
    # urls_search: list[tuple] = UDNCrawler.get_urls_search(keywords, 10)
    
    # crawler = UDNCrawler()
    # headlines = crawler.get_headline(keywords, 1)

    # for headline in headlines:
    #     print(headline.url)
    #     news = UDNCrawler.fetch_and_save_news(headline.url, keywords)
    #     if UDNCrawler.skipped:
    #         print(f"Skipped {headline.url}:")
    #         continue
    #     rand_sleep = 1.0 + random.random() * 3.0
    #     time.sleep(rand_sleep)




        
    # for name, url in urls_instant:
        # news = UDNCrawler.fetch_and_save_news(url, UDNCategorys.INSTANT.name)

    # test_news = NewsWithSummary(
    #     title="Title",
    #     time="Time",
    #     url="URL",
    #     content="Content",
    #     summary="Summary",
    #     reason="Reason"
    # )

    # test_news = UDNCrawler.fetch_and_save_news("https://money.udn.com/money/story/12040/8389105", "TEST")
    # print(test_news, end="")


if __name__ == "__main__":
    main()