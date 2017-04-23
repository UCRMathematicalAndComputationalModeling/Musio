# swagger_client.AlbumApi

All URIs are relative to *https://api.musixmatch.com/ws/1.1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**album_get_get**](AlbumApi.md#album_get_get) | **GET** /album.get | 
[**artist_albums_get_get**](AlbumApi.md#artist_albums_get_get) | **GET** /artist.albums.get | 


# **album_get_get**
> InlineResponse200 album_get_get(album_id, format=format, callback=callback)





### Example 
```python
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: key
swagger_client.configuration.api_key['apikey'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# swagger_client.configuration.api_key_prefix['apikey'] = 'Bearer'

# create an instance of the API class
api_instance = swagger_client.AlbumApi()
album_id = 'album_id_example' # str | The musiXmatch album id
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)

try: 
    # 
    api_response = api_instance.album_get_get(album_id, format=format, callback=callback)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling AlbumApi->album_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **album_id** | **str**| The musiXmatch album id | 
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 

### Return type

[**InlineResponse200**](InlineResponse200.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **artist_albums_get_get**
> InlineResponse2002 artist_albums_get_get(artist_id, format=format, callback=callback, s_release_date=s_release_date, g_album_name=g_album_name, page_size=page_size, page=page)





### Example 
```python
import time
import swagger_client
from swagger_client.rest import ApiException
from pprint import pprint

# Configure API key authorization: key
swagger_client.configuration.api_key['apikey'] = 'YOUR_API_KEY'
# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
# swagger_client.configuration.api_key_prefix['apikey'] = 'Bearer'

# create an instance of the API class
api_instance = swagger_client.AlbumApi()
artist_id = 'artist_id_example' # str | The musiXmatch artist id
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)
s_release_date = 's_release_date_example' # str | Sort by release date (asc|desc) (optional)
g_album_name = 'g_album_name_example' # str | Group by Album Name (optional)
page_size = 3.4 # float | Define the page size for paginated results.Range is 1 to 100. (optional)
page = 3.4 # float | Define the page number for paginated results (optional)

try: 
    # 
    api_response = api_instance.artist_albums_get_get(artist_id, format=format, callback=callback, s_release_date=s_release_date, g_album_name=g_album_name, page_size=page_size, page=page)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling AlbumApi->artist_albums_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **artist_id** | **str**| The musiXmatch artist id | 
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 
 **s_release_date** | **str**| Sort by release date (asc|desc) | [optional] 
 **g_album_name** | **str**| Group by Album Name | [optional] 
 **page_size** | **float**| Define the page size for paginated results.Range is 1 to 100. | [optional] 
 **page** | **float**| Define the page number for paginated results | [optional] 

### Return type

[**InlineResponse2002**](InlineResponse2002.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

