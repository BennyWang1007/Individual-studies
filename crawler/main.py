import random
import time
from tqdm import tqdm

from .udn_crawler import UDNCategory, UDNCategorys, UDNCrawler
from .utils import Logger

cur_page = 155


def test_get_news_with_category(
    catagory: UDNCategory,
    num: int = 10,
    logger: Logger = Logger("__main__")
) -> None:

    global cur_page
    logger.info(f"Start to get {num} news from {catagory.name}")
    count = 0

    with tqdm(
        total=num,
        desc=f"Get {count} news from {catagory.name}"
    ) as pbar:
        while count < num:
            headlines, next_page = UDNCrawler.get_headlines_catagory(
                catagory, num - count, page=cur_page
            )
            for headline in headlines:
                _ = UDNCrawler.fetch_and_save_news(
                    headline.url, catagory.name, skip_if_crawled=True
                )
                if UDNCrawler.skipped:
                    logger.info(f"Skipped {headline.title}, {headline.url}:")
                    continue
                count += 1
                pbar.update(1)
                rand_sleep = 0.1 + random.random() * 0.1
                time.sleep(rand_sleep)

            if next_page is not None:
                cur_page = next_page
            # rand_sleep = 0.5 + random.random() * 0.5
            # time.sleep(rand_sleep)
    logger.info(f"Get {count} news from {catagory.name}")


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


def main() -> None:
    UDNCrawler()
    # suppress the logger.info
    UDNCrawler.logger.set_verbose_level(0)
    test_get_news_with_category(UDNCategorys.INSTANT, 200)


if __name__ == "__main__":
    main()
