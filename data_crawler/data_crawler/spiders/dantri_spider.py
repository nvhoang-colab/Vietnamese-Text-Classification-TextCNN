from scrapy import Spider
from scrapy.selector import Selector
from data_crawler.items import CrawlerItem
import pandas as pd

category_list = ['Kinh doanh',
                 'Pháp luật',
                 'Thể thao',
                 'Sức khỏe',
                 'Giải trí',
                 'Giáo dục']

path = 'url_Dantri.csv'

def read_urls(path):
    df = pd.read_csv(path, index_col='category').dropna()
    all_url = df.transpose().to_dict()
    print(all_url.keys())
    urls = ["https://beta.dantri.com.vn" + all_url[cate]['url'].strip() for cate in category_list]
    return urls

def url_gen(urls):
    res = []
    page_from = 1
    page_to = 100
    for i in range(page_from, page_to):
        for url in urls:
            url = url.replace('.htm', '/trang-'+ str(i) + '.htm')
            res.append(url)
    return res

class ZingnewsSpider(Spider):
    name = "Dantri"
    allowed_domains = ["https://beta.dantri.com.vn"]
    urls = read_urls('url_Dantri.csv')
    start_urls = url_gen(urls)

    def parse(self, response):
        category = response.xpath(
            '//h1[@class="page-header__title"]/a/@title').extract_first()
        for article in response.xpath('//ul[@class="dt-list dt-list--lg"]/li'):
            item = CrawlerItem()
            item['title'] = article.xpath(
                'div/div/h3/a/text()').extract_first()
            item['description'] = article.xpath(
                'div/div/a/text()').extract_first()
            item['category'] = category.lower()
            yield item
