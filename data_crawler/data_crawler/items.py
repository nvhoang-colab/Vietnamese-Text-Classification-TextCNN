# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
class CrawlerItem(scrapy.Item):
    title = scrapy.Field()
    description = scrapy.Field()
    category = scrapy.Field()
    pass

class UrlCrawlerItem(scrapy.Item):
    category = scrapy.Field()
    url = scrapy.Field()
    pass