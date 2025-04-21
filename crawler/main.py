import random
import time
from tqdm import tqdm

from .udn_crawler import UDNCategory, UDNCategorys, UDNCrawler
from .utils import Logger

MAX_NEWS_PER_FETCH = 100

cur_page = 1


def get_news_with_category(
    catagory: UDNCategory,
    num: int = 10,
    start_page: int = -1,
    logger: Logger = Logger("__main__")
) -> None:

    global cur_page

    if start_page > 0:
        cur_page = start_page

    while num > MAX_NEWS_PER_FETCH:
        get_news_with_category(catagory, MAX_NEWS_PER_FETCH, logger=logger)
        num -= MAX_NEWS_PER_FETCH

    logger.info(f"Start to get {num} news from {catagory.name}")

    with tqdm(
        total=num, desc=f"Get {num} news from {catagory.name}"
    ) as pbar:
        while num > 0:
            headlines, next_page = UDNCrawler.get_headlines_catagory(
                catagory, num, page=cur_page
            )
            logger.debug(
                f"Get {len(headlines)} news from {catagory.name} "
                f"page {cur_page}-{next_page}"
            )

            unfetched_urls = set([headline.url for headline in headlines])
            unfetched_urls -= UDNCrawler.crawled_urls_only
            unfetched_urls -= UDNCrawler.crawled_failed_urls
            for unfetched_url in unfetched_urls:
                _ = UDNCrawler.fetch_and_save_news(
                    # since we have filter out the crawled urls
                    unfetched_url, catagory.name, skip_if_crawled=False
                )
                if UDNCrawler.skipped:
                    # logger.info(f"Skipped {headline.title}, {headline.url}:")
                    continue
                num -= 1
                pbar.update(1)
                rand_sleep = (0.1 + random.random() * 0.1) / 2
                time.sleep(rand_sleep)

            if next_page is not None:
                cur_page = next_page


def test_get_news_with_keywords(keywords: str, num: int = 10) -> None:
    page = 1
    headlines = UDNCrawler.get_headlines_keyword(keywords, page=page)
    for headline in headlines:
        _ = UDNCrawler.fetch_and_save_news(headline.url, keywords)
        if UDNCrawler.skipped:
            print(f"Skipped {headline.title}, {headline.url}:")
            continue
        rand_sleep = 1.0 + random.random() * 1.0
        time.sleep(rand_sleep)


def crawler_main(num: int = 10, start_page: int = 1) -> None:
    # init the crawler
    UDNCrawler()
    UDNCrawler.logger.set_verbose_level(1)  # suppress the info log

    crawler_logger = Logger("udn_crawler")
    crawler_logger.set_verbose_level(4)  # to recieve debug log

    get_news_with_category(
        UDNCategorys.INSTANT, num, start_page, crawler_logger
    )


if __name__ == "__main__":
    crawler_main()
