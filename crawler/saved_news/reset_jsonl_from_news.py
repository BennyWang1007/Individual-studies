import json

if __name__ == "__main__":
    # Reset the crawled urls-only file
    with open("crawled_urls_only.json", "w", encoding="utf-8") as f:
        pass

    # Reset the crawled urls file
    with open("crawled_urls.json", "w", encoding="utf-8") as f:
        pass

    crawled_urls_only: list[str] = []
    crawled_urls: list[tuple[str, str]] = []
    news_list: set[str] = set()

    # Read the crawled urls from the udn_news.jsonl file
    with open("udn_news.jsonl", "r", encoding="utf-8") as f:
        id = 0
        for line in f:
            data = json.loads(line)
            if data["content"] in news_list:
                print(f"Duplicated news id: {id}")
            news_list.add(data["content"])
            crawled_urls_only.append(data["url"])
            crawled_urls.append(
                (data["url"], "crawler/saved_news/udn_news.jsonl")
            )
            id += 1

    # Write the crawled urls to the crawled_urls_only.jsonl file
    with open("crawled_urls_only.json", "w", encoding="utf-8") as f:
        json.dump(crawled_urls_only, f, ensure_ascii=False, indent=4)

    # Write the crawled urls to the crawled_urls.jsonl file
    with open("crawled_urls.json", "w", encoding="utf-8") as f:
        json.dump(crawled_urls, f, ensure_ascii=False, indent=4)

    print(f"Total crawled urls: {len(crawled_urls_only)}")
