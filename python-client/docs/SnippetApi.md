# swagger_client.SnippetApi

All URIs are relative to *https://api.musixmatch.com/ws/1.1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**track_snippet_get_get**](SnippetApi.md#track_snippet_get_get) | **GET** /track.snippet.get | 


# **track_snippet_get_get**
> InlineResponse20010 track_snippet_get_get(track_id, format=format, callback=callback)





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
api_instance = swagger_client.SnippetApi()
track_id = 'track_id_example' # str | The musiXmatch track id
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)

try: 
    # 
    api_response = api_instance.track_snippet_get_get(track_id, format=format, callback=callback)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling SnippetApi->track_snippet_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **track_id** | **str**| The musiXmatch track id | 
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 

### Return type

[**InlineResponse20010**](InlineResponse20010.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

