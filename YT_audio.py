from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class Music:
    def __init__(self):
        # Initialize the Chrome WebDriver
        # Make sure you have the correct version of ChromeDriver installed
        self.driver = webdriver.Chrome()

    def play(self, query):
        try:
            # Open YouTube
            self.driver.get("https://www.youtube.com")

            # Wait for the search bar to load
            WebDriverWait(self.driver, timeout=10).until(
                EC.presence_of_element_located((By.NAME, "search_query"))
            )

            # Find the search bar and input the query
            search_box = self.driver.find_element(By.NAME, "search_query")
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)

            # Wait for search results to load
            WebDriverWait(self.driver, timeout=10).until(
                EC.presence_of_element_located((By.XPATH, '//a[@id="video-title"]'))
            )

            # Click on the first video
            video = self.driver.find_element(By.XPATH, '(//a[@id="video-title"])[1]')
            video.click()

            print("Playing the video on YouTube...")

        except Exception as e:
            print(f"An error occurred: {e}")
            self.driver.quit()

    def quit_driver(self):
        # Close the browser
        self.driver.quit()