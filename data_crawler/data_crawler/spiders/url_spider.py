from scrapy import Spider
from scrapy.selector import Selector
from data_crawler.items import CrawlerItem
from data_crawler.items import UrlCrawlerItem

class ZingnewsSpider(Spider):
    name = "Url"
    allowed_domains = ["https://vnexpress.net/"]
    start_urls = ["https://vnexpress.net/"]
    def parse(self, response):
        for li in response.xpath('//ul[@class="parent"]/li'):
            item = UrlCrawlerItem()
            item['url'] = li.xpath(
                'a/@href').extract_first()
            item['category'] = li.xpath(
                'a/@title').extract_first()
            yield item
