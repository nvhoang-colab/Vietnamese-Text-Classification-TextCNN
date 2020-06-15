from scrapy import Spider
from scrapy.selector import Selector
from data_crawler.items import CrawlerItem

def urls(page_from=1, page_to=10):
    res = []
    for i in range(page_from, page_to):
        res.append('https://news.zing.vn/suc-khoe/trang' + str(i) + '.html')
    return res

class ZingnewsSpider(Spider):
    name = "Zingnews"
    allowed_domains = ["https://zingnews.vn/"]
    start_urls = urls(1,51)

    def parse(self, response):
        for article in response.xpath("//div/article"):
            item = CrawlerItem()

            item['title'] = article.xpath(
                'header/p[@class="article-title"]/a/text()').extract_first()
            item['description'] = article.xpath(
                'header/p[@class="article-summary"]/text()').extract_first()
            item['category'] = 'sức khỏe'

            yield item
