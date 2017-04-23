# swagger_client.LyricsApi

All URIs are relative to *https://api.musixmatch.com/ws/1.1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**matcher_lyrics_get_get**](LyricsApi.md#matcher_lyrics_get_get) | **GET** /matcher.lyrics.get | 
[**track_lyrics_get_get**](LyricsApi.md#track_lyrics_get_get) | **GET** /track.lyrics.get | 


# **matcher_lyrics_get_get**
> InlineResponse2007 matcher_lyrics_get_get(format=format, callback=callback, q_track=q_track, q_artist=q_artist)





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
api_instance = swagger_client.LyricsApi()
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)
q_track = 'q_track_example' # str | The song title (optional)
q_artist = 'q_artist_example' # str | The song artist (optional)

try: 
    # 
    api_response = api_instance.matcher_lyrics_get_get(format=format, callback=callback, q_track=q_track, q_artist=q_artist)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling LyricsApi->matcher_lyrics_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 
 **q_track** | **str**| The song title | [optional] 
 **q_artist** | **str**| The song artist | [optional] 

### Return type

[**InlineResponse2007**](InlineResponse2007.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **track_lyrics_get_get**
> InlineResponse2007 track_lyrics_get_get(track_id, format=format, callback=callback)





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
api_instance = swagger_client.LyricsApi()
track_id = 'track_id_example' # str | The musiXmatch track id
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)

try: 
    # 
    api_response = api_instance.track_lyrics_get_get(track_id, format=format, callback=callback)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling LyricsApi->track_lyrics_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **track_id** | **str**| The musiXmatch track id | 
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 

### Return type

[**InlineResponse2007**](InlineResponse2007.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

