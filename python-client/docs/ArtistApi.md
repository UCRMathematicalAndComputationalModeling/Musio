# swagger_client.ArtistApi

All URIs are relative to *https://api.musixmatch.com/ws/1.1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**artist_get_get**](ArtistApi.md#artist_get_get) | **GET** /artist.get | 
[**artist_related_get_get**](ArtistApi.md#artist_related_get_get) | **GET** /artist.related.get | 
[**artist_search_get**](ArtistApi.md#artist_search_get) | **GET** /artist.search | 
[**chart_artists_get_get**](ArtistApi.md#chart_artists_get_get) | **GET** /chart.artists.get | 


# **artist_get_get**
> InlineResponse2003 artist_get_get(artist_id, format=format, callback=callback)





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
api_instance = swagger_client.ArtistApi()
artist_id = 'artist_id_example' # str |  The musiXmatch artist id
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)

try: 
    # 
    api_response = api_instance.artist_get_get(artist_id, format=format, callback=callback)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling ArtistApi->artist_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **artist_id** | **str**|  The musiXmatch artist id | 
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 

### Return type

[**InlineResponse2003**](InlineResponse2003.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **artist_related_get_get**
> InlineResponse2004 artist_related_get_get(artist_id, format=format, callback=callback, page_size=page_size, page=page)





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
api_instance = swagger_client.ArtistApi()
artist_id = 'artist_id_example' # str | The musiXmatch artist id
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)
page_size = 3.4 # float | Define the page size for paginated results.Range is 1 to 100. (optional)
page = 3.4 # float | Define the page number for paginated results (optional)

try: 
    # 
    api_response = api_instance.artist_related_get_get(artist_id, format=format, callback=callback, page_size=page_size, page=page)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling ArtistApi->artist_related_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **artist_id** | **str**| The musiXmatch artist id | 
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 
 **page_size** | **float**| Define the page size for paginated results.Range is 1 to 100. | [optional] 
 **page** | **float**| Define the page number for paginated results | [optional] 

### Return type

[**InlineResponse2004**](InlineResponse2004.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **artist_search_get**
> InlineResponse2004 artist_search_get(format=format, callback=callback, q_artist=q_artist, f_artist_id=f_artist_id, page=page, page_size=page_size)





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
api_instance = swagger_client.ArtistApi()
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)
q_artist = 'q_artist_example' # str | The song artist (optional)
f_artist_id = 3.4 # float | When set, filter by this artist id (optional)
page = 3.4 # float | Define the page number for paginated results (optional)
page_size = 3.4 # float | Define the page size for paginated results.Range is 1 to 100. (optional)

try: 
    # 
    api_response = api_instance.artist_search_get(format=format, callback=callback, q_artist=q_artist, f_artist_id=f_artist_id, page=page, page_size=page_size)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling ArtistApi->artist_search_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 
 **q_artist** | **str**| The song artist | [optional] 
 **f_artist_id** | **float**| When set, filter by this artist id | [optional] 
 **page** | **float**| Define the page number for paginated results | [optional] 
 **page_size** | **float**| Define the page size for paginated results.Range is 1 to 100. | [optional] 

### Return type

[**InlineResponse2004**](InlineResponse2004.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **chart_artists_get_get**
> InlineResponse2005 chart_artists_get_get(format=format, callback=callback, page=page, page_size=page_size, country=country)





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
api_instance = swagger_client.ArtistApi()
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)
page = 3.4 # float | Define the page number for paginated results (optional)
page_size = 3.4 # float | Define the page size for paginated results.Range is 1 to 100. (optional)
country = 'us' # str | A valid ISO 3166 country code (optional) (default to us)

try: 
    # 
    api_response = api_instance.chart_artists_get_get(format=format, callback=callback, page=page, page_size=page_size, country=country)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling ArtistApi->chart_artists_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 
 **page** | **float**| Define the page number for paginated results | [optional] 
 **page_size** | **float**| Define the page size for paginated results.Range is 1 to 100. | [optional] 
 **country** | **str**| A valid ISO 3166 country code | [optional] [default to us]

### Return type

[**InlineResponse2005**](InlineResponse2005.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

