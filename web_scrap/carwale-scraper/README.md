## Introduction

Web scrapping or crawling is the process of collecting data from a website's source code. A scraper accesses the source code of a website using the Hypertext Transfer Protocol or through a web browser.

![](https://github.com/deveshdatwani/Carwale-Scraper/blob/master/scrape.jpg)

## The Goal

The aim of this project is to experiment with web scraping. For this we use Python and its libraries such as requests and BeautifulSoup.

We shall scrape the very prominent carwale.com to retrieve user reviews of cars enlisted on it.

![](https://github.com/deveshdatwani/Carwale-Scraper/blob/master/carwale.PNG)

carwale.com contains car review data from thousands of users accross India. Scraping the reviews can give us a brief public perception about any car based on its reviews.

## Technology Used

* Python 
* Requests
* BeautifulSoup

## Installation and Launch (Linux)

Clone the project repo using the git clone command

``` 
git clone https://github.com/deveshdatwani/Carwale-Scraper.git
```

Launch scraper by running the Car_Wale_Review.py module 

```
python3 Car_Wale_Review.py
```

You will see this screen subsequently (if the website source code hasn't been altered)

![](https://github.com/deveshdatwani/Carwale-Scraper/blob/master/scraper-car.png)

You can now navigate through the website and go through any car's reviews using the corresponding numbers against automakers/cars.

## Project Walkthrough

The requests library requests the html page from carwale.com. One important thing here is to add headers to your reuquest. Many websites refuse to respond to requests arising without headers so as to prevent its website from being scrapped. The header show below fools the website servers in believing that a Mozilla Firefox browser is making this request.

```
#Fetching html of main page as well as creating a list that will be filled with automaker names along with indexes for user control
main_url = 'https://www.carwale.com'
headers = {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36'}
response = requests.get(main_url, headers=headers).text
bs_object = Bsoup(response, 'lxml')
brand_list = []
```

The response is converted into a BeautifulSoup Object. The BeautifulSoup package "beautifully" parses the response into a data structure that can be easily navigated. Go through its documnetation.

Now that we have a soup object. We can navigate through the website's source code (HTML) and locate required tags and their content using methods find/find_all.

We can save this data in .txt format or simply run sentimental analysis on it.

## Scope

The next step is to run sentimental analysis on the collected review-data. Stoping here for now. Will get back once I learn Natural Language Processing.

Ciao!
