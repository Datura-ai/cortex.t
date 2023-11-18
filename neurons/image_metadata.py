from PIL import Image
import requests
from io import BytesIO

# URL of the generated image
image_url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-qCq4dk0xvMql4Wo8WT2B9QAv/user-ull71ynh86YyNbnhwT4G04KN/img-HeTVjXC7bZ5lRk4l9yKBYHHk.png?st=2023-11-18T02%3A59%3A54Z&se=2023-11-18T04%3A59%3A54Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-11-17T19%3A55%3A58Z&ske=2023-11-18T19%3A55%3A58Z&sks=b&skv=2021-08-06&sig=WXE4/xVnsNiEt%2BtS8oh9CmyQRqNUCAU3OC8eNQCn4NE%3D"

# Download the image
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# Print image metadata (EXIF, etc.)
print(img.info)