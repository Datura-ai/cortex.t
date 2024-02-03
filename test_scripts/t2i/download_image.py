import requests

# URL of the image
url = 'https://file.io/a9znfUZ9uKvR'

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Open a file in binary write mode
    with open('image.jpg', 'wb') as file:
        # Write the content of the response to the file
        file.write(response.content)
    print("Image downloaded and saved as image.jpg")
else:
    print("Failed to download the image. Status code:", response.status_code)