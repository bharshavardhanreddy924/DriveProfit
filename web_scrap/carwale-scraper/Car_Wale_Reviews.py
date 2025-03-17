import requests
import lxml
from bs4 import BeautifulSoup as Bsoup
from fake_useragent import UserAgent
import time
import Sentimental_Analysis
from matplotlib import pyplot as plt

#Fetching html of main page as well as creating a list that will be filled with automaker names along with indexes for user control
main_url = 'https://www.cardekho.com/mercedes-benz/gla/user-reviews'
headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}
response = requests.get(main_url, headers=headers).text
bs_object = Bsoup(response, 'lxml')
brand_list = []

#Creating a loop to retrieve reviews of multiple automakers
while True:

#Finding all the automakers enlisted on Carwale
	automakers = bs_object.find(string='View More Brands').parent.parent
	automakers = automakers.find_all('li')
	for i, automaker in enumerate(automakers):
		print(i,automaker.text)
		brand_list.append(automaker)

#Allowing user to choose from the enlisted automakers & fetching car models enlisted by the automaker
	select_automaker = int(input('Select automaker index'))
	automaker_webpage = main_url + brand_list[select_automaker].a['href']
	automaker_webpage_response = requests.get(automaker_webpage, headers=headers).text
	automaker_webpage_response = Bsoup(automaker_webpage_response, 'lxml')
	cars = automaker_webpage_response.find(class_='_3B0ikd').div.ul.find_all('li')
	print(len(cars))
	car_list = []
	for i, car in enumerate(cars):
		car_list.append((i, car.find('h2').text, car.find('h2').parent['href']))

	for car in car_list:
		print(car[0], car[1], car[2])
	print('')

#Directing to the page of the selected car model and retrieving the url of the page its user reviews
	car_name = int(input('Select car index from above: '))
	car_review_page_url = main_url + car_list[car_name][2] + 'user-reviews/'
	print(car_review_page_url)

#Scraping incomplete reviews on the main page of user reviews. For complete review, each review url has to be fetched and the page scraped
	car_reviews_page = requests.get(car_review_page_url, headers = headers).text
	car_review_page = Bsoup(car_reviews_page, 'lxml')
	reviews = car_review_page.find_all(string='Read More')
	for i in reviews:
		print(i.parent.parent.text)



