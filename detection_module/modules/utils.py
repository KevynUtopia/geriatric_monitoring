import easyocr


def extract_timestamp(image):
    reader = easyocr.Reader(['en'])
    timestamp = reader.readtext(image[:200, 800:1200], detail = 0)[0]
    timestamp = ''.join([char for char in timestamp if char.isdigit()])
    if len(timestamp) == 6:
        return timestamp
    return None