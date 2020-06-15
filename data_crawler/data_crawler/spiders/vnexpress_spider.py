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

path = 'url_VNexpress.csv'

def read_urls(path):
    df = pd.read_csv(path, index_col='category').dropna()
    all_url = df.transpose().to_dict()
    print(all_url.keys())
    urls = []
    for cate in category_list:
        if cate not in ['Pháp luật', 'Giáo dục']:
            urls.append("https://vnexpress.net" + all_url[cate]['url'].strip() + '/p')
        else:
            urls.append("https://vnexpress.net" + all_url[cate]['url'].strip() + '-p')
    return urls

def url_gen(urls):
    res = []
    page_from = 1
    page_to = 100
    for i in range(page_from, page_to):
        for url in urls:
            res.append(url + str(i))
    return res

class ZingnewsSpider(Spider):
    name = "VNexpress"
    allowed_domains = ["https://vnexpress.net/"]
    urls = read_urls(path)
    start_urls = url_gen(urls)

    def parse(self, response):
        category = response.xpath(
            '//div[@class="title-folder"]/h1/a/text()').extract_first()
        for article in response.xpath("//div/article"):
            item = CrawlerItem()
            title = article.xpath(
                'h2[@class="title-news"]/a/text()').extract_first()
            if title == None:
                title = article.xpath(
                    'h3[@class="title-news"]/a/text()').extract_first()
            item['title'] = title

            item['description'] = article.xpath(
                'p[@class="description"]/a/text()').extract_first()  

            item['category'] = category         
            yield item
