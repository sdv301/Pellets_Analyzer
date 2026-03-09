from main import app
import traceback

with app.test_client() as client:
    try:
        print("GET /ml_dashboard ...")
        response = client.get('/ml_dashboard')
        print(f"Status Code: {response.status_code}")
        if response.status_code == 500:
            print("Error 500 detected. Details follows:")
            # We can't easily get the traceback from the response body if it's not custom,
            # but if we run this with app.debug = True, it might work or we can catch it here.
            print(response.data.decode('utf-8')[:2000])
    except Exception as e:
        traceback.print_exc()
