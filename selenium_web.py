from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys


class Info:
    def __init__(self):
        # Initialize the Chrome WebDriver
        self.driver = webdriver.Chrome()

    def get_info(self, query):
        # Open Wikipedia
        self.driver.get("https://www.wikipedia.org")

        # Find the search input box
        search = self.driver.find_element(By.ID, "searchInput")
        search.send_keys(query)  # Enter the query
        search.send_keys(Keys.RETURN)  # Press Enter to search

        # Wait for the user to close manually
        input("Press Enter to close the browser...")  # Keeps the browser open

        # Quit the browser
        self.driver.quit()
