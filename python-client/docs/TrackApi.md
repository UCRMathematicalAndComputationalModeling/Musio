# swagger_client.TrackApi

All URIs are relative to *https://api.musixmatch.com/ws/1.1*

Method | HTTP request | Description
------------- | ------------- | -------------
[**album_tracks_get_get**](TrackApi.md#album_tracks_get_get) | **GET** /album.tracks.get | 
[**chart_tracks_get_get**](TrackApi.md#chart_tracks_get_get) | **GET** /chart.tracks.get | 
[**matcher_track_get_get**](TrackApi.md#matcher_track_get_get) | **GET** /matcher.track.get | 
[**track_get_get**](TrackApi.md#track_get_get) | **GET** /track.get | 
[**track_search_get**](TrackApi.md#track_search_get) | **GET** /track.search | 


# **album_tracks_get_get**
> InlineResponse2001 album_tracks_get_get(album_id, format=format, callback=callback, f_has_lyrics=f_has_lyrics, page=page, page_size=page_size)





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
api_instance = swagger_client.TrackApi()
album_id = 'album_id_example' # str | The musiXmatch album id
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)
f_has_lyrics = 'f_has_lyrics_example' # str | When set, filter only contents with lyrics (optional)
page = 3.4 # float | Define the page number for paginated results (optional)
page_size = 3.4 # float | Define the page size for paginated results.Range is 1 to 100. (optional)

try: 
    # 
    api_response = api_instance.album_tracks_get_get(album_id, format=format, callback=callback, f_has_lyrics=f_has_lyrics, page=page, page_size=page_size)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling TrackApi->album_tracks_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **album_id** | **str**| The musiXmatch album id | 
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 
 **f_has_lyrics** | **str**| When set, filter only contents with lyrics | [optional] 
 **page** | **float**| Define the page number for paginated results | [optional] 
 **page_size** | **float**| Define the page size for paginated results.Range is 1 to 100. | [optional] 

### Return type

[**InlineResponse2001**](InlineResponse2001.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **chart_tracks_get_get**
> InlineResponse2006 chart_tracks_get_get(format=format, callback=callback, page=page, page_size=page_size, country=country, f_has_lyrics=f_has_lyrics)





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
api_instance = swagger_client.TrackApi()
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)
page = 3.4 # float | Define the page number for paginated results (optional)
page_size = 3.4 # float | Define the page size for paginated results.Range is 1 to 100. (optional)
country = 'us' # str | A valid ISO 3166 country code (optional) (default to us)
f_has_lyrics = 'f_has_lyrics_example' # str | When set, filter only contents with lyrics (optional)

try: 
    # 
    api_response = api_instance.chart_tracks_get_get(format=format, callback=callback, page=page, page_size=page_size, country=country, f_has_lyrics=f_has_lyrics)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling TrackApi->chart_tracks_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 
 **page** | **float**| Define the page number for paginated results | [optional] 
 **page_size** | **float**| Define the page size for paginated results.Range is 1 to 100. | [optional] 
 **country** | **str**| A valid ISO 3166 country code | [optional] [default to us]
 **f_has_lyrics** | **str**| When set, filter only contents with lyrics | [optional] 

### Return type

[**InlineResponse2006**](InlineResponse2006.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **matcher_track_get_get**
> InlineResponse2009 matcher_track_get_get(format=format, callback=callback, q_artist=q_artist, q_track=q_track, f_has_lyrics=f_has_lyrics, f_has_subtitle=f_has_subtitle)





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
api_instance = swagger_client.TrackApi()
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)
q_artist = 'q_artist_example' # str | The song artist (optional)
q_track = 'q_track_example' # str | The song title (optional)
f_has_lyrics = 3.4 # float | When set, filter only contents with lyrics (optional)
f_has_subtitle = 3.4 # float | When set, filter only contents with subtitles (optional)

try: 
    # 
    api_response = api_instance.matcher_track_get_get(format=format, callback=callback, q_artist=q_artist, q_track=q_track, f_has_lyrics=f_has_lyrics, f_has_subtitle=f_has_subtitle)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling TrackApi->matcher_track_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 
 **q_artist** | **str**| The song artist | [optional] 
 **q_track** | **str**| The song title | [optional] 
 **f_has_lyrics** | **float**| When set, filter only contents with lyrics | [optional] 
 **f_has_subtitle** | **float**| When set, filter only contents with subtitles | [optional] 

### Return type

[**InlineResponse2009**](InlineResponse2009.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **track_get_get**
> InlineResponse2009 track_get_get(track_id, format=format, callback=callback)





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
api_instance = swagger_client.TrackApi()
track_id = 'track_id_example' # str | The musiXmatch track id
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)

try: 
    # 
    api_response = api_instance.track_get_get(track_id, format=format, callback=callback)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling TrackApi->track_get_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **track_id** | **str**| The musiXmatch track id | 
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 

### Return type

[**InlineResponse2009**](InlineResponse2009.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

# **track_search_get**
> InlineResponse2006 track_search_get(format=format, callback=callback, q_track=q_track, q_artist=q_artist, q_lyrics=q_lyrics, f_artist_id=f_artist_id, f_music_genre_id=f_music_genre_id, f_lyrics_language=f_lyrics_language, f_has_lyrics=f_has_lyrics, s_artist_rating=s_artist_rating, s_track_rating=s_track_rating, quorum_factor=quorum_factor, page_size=page_size, page=page)





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
api_instance = swagger_client.TrackApi()
format = 'json' # str | output format: json, jsonp, xml. (optional) (default to json)
callback = 'callback_example' # str | jsonp callback (optional)
q_track = 'q_track_example' # str | The song title (optional)
q_artist = 'q_artist_example' # str | The song artist (optional)
q_lyrics = 'q_lyrics_example' # str | Any word in the lyrics (optional)
f_artist_id = 3.4 # float | When set, filter by this artist id (optional)
f_music_genre_id = 3.4 # float | When set, filter by this music category id (optional)
f_lyrics_language = 3.4 # float | Filter by the lyrics language (en,it,..) (optional)
f_has_lyrics = 3.4 # float | When set, filter only contents with lyrics (optional)
s_artist_rating = 's_artist_rating_example' # str | Sort by our popularity index for artists (asc|desc) (optional)
s_track_rating = 's_track_rating_example' # str | Sort by our popularity index for tracks (asc|desc) (optional)
quorum_factor = 1 # float | Search only a part of the given query string.Allowed range is (0.1 – 0.9) (optional) (default to 1)
page_size = 3.4 # float | Define the page size for paginated results.Range is 1 to 100. (optional)
page = 3.4 # float | Define the page number for paginated results (optional)

try: 
    # 
    api_response = api_instance.track_search_get(format=format, callback=callback, q_track=q_track, q_artist=q_artist, q_lyrics=q_lyrics, f_artist_id=f_artist_id, f_music_genre_id=f_music_genre_id, f_lyrics_language=f_lyrics_language, f_has_lyrics=f_has_lyrics, s_artist_rating=s_artist_rating, s_track_rating=s_track_rating, quorum_factor=quorum_factor, page_size=page_size, page=page)
    pprint(api_response)
except ApiException as e:
    print "Exception when calling TrackApi->track_search_get: %s\n" % e
```

### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
 **format** | **str**| output format: json, jsonp, xml. | [optional] [default to json]
 **callback** | **str**| jsonp callback | [optional] 
 **q_track** | **str**| The song title | [optional] 
 **q_artist** | **str**| The song artist | [optional] 
 **q_lyrics** | **str**| Any word in the lyrics | [optional] 
 **f_artist_id** | **float**| When set, filter by this artist id | [optional] 
 **f_music_genre_id** | **float**| When set, filter by this music category id | [optional] 
 **f_lyrics_language** | **float**| Filter by the lyrics language (en,it,..) | [optional] 
 **f_has_lyrics** | **float**| When set, filter only contents with lyrics | [optional] 
 **s_artist_rating** | **str**| Sort by our popularity index for artists (asc|desc) | [optional] 
 **s_track_rating** | **str**| Sort by our popularity index for tracks (asc|desc) | [optional] 
 **quorum_factor** | **float**| Search only a part of the given query string.Allowed range is (0.1 – 0.9) | [optional] [default to 1]
 **page_size** | **float**| Define the page size for paginated results.Range is 1 to 100. | [optional] 
 **page** | **float**| Define the page number for paginated results | [optional] 

### Return type

[**InlineResponse2006**](InlineResponse2006.md)

### Authorization

[key](../README.md#key)

### HTTP request headers

 - **Content-Type**: application/json
 - **Accept**: application/json

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

