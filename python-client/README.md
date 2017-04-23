# swagger_client
Musixmatch lyrics API is a robust service that permits you to search and retrieve lyrics in the simplest possible way. It just works.  Include millions of licensed lyrics on your website or in your application legally.  The fastest, most powerful and legal way to display lyrics on your website or in your application.  #### Read musixmatch API Terms & Conditions and the Privacy Policy: Before getting started, you must take a look at the [API Terms & Conditions](http://musixmatch.com/apiterms/) and the [Privacy Policy](https://developer.musixmatch.com/privacy). We’ve worked hard to make this service completely legal so that we are all protected from any foreseeable liability. Take the time to read this stuff.  #### Register for an API key: All you need to do is [register](https://developer.musixmatch.com/signup) in order to get your API key, a mandatory parameter for most of our API calls. It’s your personal identifier and should be kept secret:  ```   https://api.musixmatch.com/ws/v1.1/track.get?apikey=YOUR_API_KEY ``` #### Integrate the musixmatch service with your web site or application In the most common scenario you only need to implement two API calls:  The first call is to match your catalog to ours using the [track.search](#!/Track/get_track_search) function and the second is to get the lyrics using the [track.lyrics.get](#!/Lyrics/get_track_lyrics_get) api. That’s it!  ## API Methods What does the musiXmatch API do?  The musiXmatch API allows you to read objects from our huge 100% licensed lyrics database.  To make your life easier we are providing you with one or more examples to show you how it could work in the wild. You’ll find both the API request and API response in all the available output formats for each API call. Follow the links below for the details.  The current API version is 1.1, the root URL is located at https://api.musixmatch.com/ws/1.1/  Supported input parameters can be found on the page [Input Parameters](https://developer.musixmatch.com/documentation/input-parameters). Use UTF-8 to encode arguments when calling API methods.  Every response includes a status_code. A list of all status codes can be consulted at [Status Codes](https://developer.musixmatch.com/documentation/status-codes).  ## Music meta data The musiXmatch api is built around lyrics, but there are many other data we provide through the api that can be used to improve every existent music service.  ## Track Inside the track object you can get the following extra information:  ### TRACK RATING  The track rating is a score 0-100 identifying how popular is a song in musixmatch.  You can use this information to sort search results, like the most popular songs of an artist, of a music genre, of a lyrics language.  ### INSTRUMENTAL AND EXPLICIT FLAGS  The instrumental flag identifies songs with music only, no lyrics.  The explicit flag identifies songs with explicit lyrics or explicit title. We're able to identify explicit words and set the flag for the most common languages.  ### FAVOURITES  How many users have this song in their list of favourites.  Can be used to sort tracks by num favourite to identify more popular tracks within a set.  ### MUSIC GENRE  The music genere of the song.  Can be used to group songs by genre, as input for similarity alghorithms, artist genre identification, navigate songs by genere, etc.  ### SONG TITLES TRANSLATIONS  The track title, as translated in different lanauages, can be used to display the right writing for a given user, example:  LIES (Bigbang) becomes 在光化門 in chinese HALLELUJAH (Bigbang) becomes ハレルヤ in japanese   ## Artist Inside the artist object you can get the following nice extra information:  ### COMMENTS AND COUNTRY  An artist comment is a short snippet of text which can be mainly used for disambiguation.  The artist country is the born country of the artist/group  There are two perfect search result if you search by artist with the keyword \"U2\". Indeed there are two distinct music groups with this same name, one is the most known irish group of Bono Vox, the other is a less popular (world wide speaking) group from Japan.  Here's how you can made use of the artist comment in your search result page:  U2 (Irish rock band) U2 (あきやまうに) You can also show the artist country for even better disambiguation:  U2 (Irish rock band) from Ireland U2 (あきやまうに) from Japan ARTIST TRANSLATIONS  When you create a world wide music related service you have to take into consideration to display the artist name in the user's local language. These translation are also used as aliases to improve the search results.  Let's use PSY for this example.  Western people know him as PSY but korean want to see the original name 싸이.  Using the name translations provided by our api you can show to every user the writing they expect to see.  Furthermore, when you search for \"psy gangnam style\" or \"싸이 gangnam style\" with our search/match api you will still be able to find the song.  ### ARTIST RATING  The artist rating is a score 0-100 identifying how popular is an artist in musixmatch.  You can use this information to build charts, for suggestions, to sort search results. In the example above about U2, we use the artist rating to show the irish band before the japanese one in our serp.  ### ARTIST MUSIC GENRE  We provide one or more main artist genre, this information can be used to calculate similar artist, suggestions, or the filter a search by artist genre.    ## Album Inside the album object you can get the following nice extra information:  ### ALBUM RATING  The album rating is a score 0-100 identifying how popular is an album in musixmatch.  You can use this information to sort search results, like the most popular albums of an artist.  ### ALBUM RATING  The album rating is a score 0-100 identifying how popular is an album in musixmatch.  You can use this information to sort search results, like the most popular albums of an artist.  ### ALBUM COPYRIGHT AND LABEL  For most of our albums we can provide extra information like for example:  Label: Universal-Island Records Ltd. Copyright: (P) 2013 Rubyworks, under license to Columbia Records, a Division of Sony Music Entertainment. ALBUM TYPE AND RELEASE DATE  The album official release date can be used to sort an artist's albums view starting by the most recent one.  Album can also be filtered or grouped by type: Single, Album, Compilation, Remix, Live. This can help to build an artist page with a more organized structure.  ### ALBUM MUSIC GENRE  For most of the albums we provide two groups of music genres. Primary and secondary. This information can be used to help user navigate albums by genre.  An example could be:  Primary genere: POP Secondary genre: K-POP or Mandopop 

This Python package is automatically generated by the [Swagger Codegen](https://github.com/swagger-api/swagger-codegen) project:

- API version: 1.1.0
- Package version: 1.0.0
- Build date: 2016-09-26T10:39:42.082Z
- Build package: class io.swagger.codegen.languages.PythonClientCodegen
For more information, please visit [https://musixmatch.com](https://musixmatch.com)

## Requirements.

Python 2.7 and 3.4+

## Installation & Usage
### pip install

If the python package is hosted on Github, you can install directly from Github

```sh
pip install git+https://github.com//.git
```
(you may need to run `pip` with root permission: `sudo pip install git+https://github.com//.git`)

Then import the package:
```python
import swagger_client 
```

### Setuptools

Install via [Setuptools](http://pypi.python.org/pypi/setuptools).

```sh
python setup.py install --user
```
(or `sudo python setup.py install` to install the package for all users)

Then import the package:
```python
import swagger_client
```

## Getting Started

Please follow the [installation procedure](#installation--usage) and then run the following:

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
api_instance = swagger_client.AlbumApi
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

## Documentation for API Endpoints

All URIs are relative to *https://api.musixmatch.com/ws/1.1*

Class | Method | HTTP request | Description
------------ | ------------- | ------------- | -------------
*AlbumApi* | [**album_get_get**](docs/AlbumApi.md#album_get_get) | **GET** /album.get | 
*AlbumApi* | [**artist_albums_get_get**](docs/AlbumApi.md#artist_albums_get_get) | **GET** /artist.albums.get | 
*ArtistApi* | [**artist_get_get**](docs/ArtistApi.md#artist_get_get) | **GET** /artist.get | 
*ArtistApi* | [**artist_related_get_get**](docs/ArtistApi.md#artist_related_get_get) | **GET** /artist.related.get | 
*ArtistApi* | [**artist_search_get**](docs/ArtistApi.md#artist_search_get) | **GET** /artist.search | 
*ArtistApi* | [**chart_artists_get_get**](docs/ArtistApi.md#chart_artists_get_get) | **GET** /chart.artists.get | 
*LyricsApi* | [**matcher_lyrics_get_get**](docs/LyricsApi.md#matcher_lyrics_get_get) | **GET** /matcher.lyrics.get | 
*LyricsApi* | [**track_lyrics_get_get**](docs/LyricsApi.md#track_lyrics_get_get) | **GET** /track.lyrics.get | 
*SnippetApi* | [**track_snippet_get_get**](docs/SnippetApi.md#track_snippet_get_get) | **GET** /track.snippet.get | 
*SubtitleApi* | [**matcher_subtitle_get_get**](docs/SubtitleApi.md#matcher_subtitle_get_get) | **GET** /matcher.subtitle.get | 
*SubtitleApi* | [**track_subtitle_get_get**](docs/SubtitleApi.md#track_subtitle_get_get) | **GET** /track.subtitle.get | 
*TrackApi* | [**album_tracks_get_get**](docs/TrackApi.md#album_tracks_get_get) | **GET** /album.tracks.get | 
*TrackApi* | [**chart_tracks_get_get**](docs/TrackApi.md#chart_tracks_get_get) | **GET** /chart.tracks.get | 
*TrackApi* | [**matcher_track_get_get**](docs/TrackApi.md#matcher_track_get_get) | **GET** /matcher.track.get | 
*TrackApi* | [**track_get_get**](docs/TrackApi.md#track_get_get) | **GET** /track.get | 
*TrackApi* | [**track_search_get**](docs/TrackApi.md#track_search_get) | **GET** /track.search | 


## Documentation For Models

 - [Album](docs/Album.md)
 - [AlbumPrimaryGenres](docs/AlbumPrimaryGenres.md)
 - [AlbumPrimaryGenresMusicGenre](docs/AlbumPrimaryGenresMusicGenre.md)
 - [AlbumPrimaryGenresMusicGenreList](docs/AlbumPrimaryGenresMusicGenreList.md)
 - [Artist](docs/Artist.md)
 - [ArtistArtistAliasList](docs/ArtistArtistAliasList.md)
 - [ArtistArtistCredits](docs/ArtistArtistCredits.md)
 - [ArtistArtistNameTranslation](docs/ArtistArtistNameTranslation.md)
 - [ArtistArtistNameTranslationList](docs/ArtistArtistNameTranslationList.md)
 - [ArtistPrimaryGenres](docs/ArtistPrimaryGenres.md)
 - [ArtistPrimaryGenresMusicGenre](docs/ArtistPrimaryGenresMusicGenre.md)
 - [ArtistPrimaryGenresMusicGenreList](docs/ArtistPrimaryGenresMusicGenreList.md)
 - [ArtistSecondaryGenres](docs/ArtistSecondaryGenres.md)
 - [InlineResponse200](docs/InlineResponse200.md)
 - [InlineResponse2001](docs/InlineResponse2001.md)
 - [InlineResponse20010](docs/InlineResponse20010.md)
 - [InlineResponse20010Message](docs/InlineResponse20010Message.md)
 - [InlineResponse20010MessageBody](docs/InlineResponse20010MessageBody.md)
 - [InlineResponse2001Message](docs/InlineResponse2001Message.md)
 - [InlineResponse2001MessageBody](docs/InlineResponse2001MessageBody.md)
 - [InlineResponse2001MessageHeader](docs/InlineResponse2001MessageHeader.md)
 - [InlineResponse2002](docs/InlineResponse2002.md)
 - [InlineResponse2002Message](docs/InlineResponse2002Message.md)
 - [InlineResponse2002MessageBody](docs/InlineResponse2002MessageBody.md)
 - [InlineResponse2002MessageHeader](docs/InlineResponse2002MessageHeader.md)
 - [InlineResponse2003](docs/InlineResponse2003.md)
 - [InlineResponse2003Message](docs/InlineResponse2003Message.md)
 - [InlineResponse2003MessageBody](docs/InlineResponse2003MessageBody.md)
 - [InlineResponse2004](docs/InlineResponse2004.md)
 - [InlineResponse2004Message](docs/InlineResponse2004Message.md)
 - [InlineResponse2004MessageBody](docs/InlineResponse2004MessageBody.md)
 - [InlineResponse2005](docs/InlineResponse2005.md)
 - [InlineResponse2005Message](docs/InlineResponse2005Message.md)
 - [InlineResponse2005MessageHeader](docs/InlineResponse2005MessageHeader.md)
 - [InlineResponse2006](docs/InlineResponse2006.md)
 - [InlineResponse2006Message](docs/InlineResponse2006Message.md)
 - [InlineResponse2006MessageBody](docs/InlineResponse2006MessageBody.md)
 - [InlineResponse2006MessageBodyTrackList](docs/InlineResponse2006MessageBodyTrackList.md)
 - [InlineResponse2007](docs/InlineResponse2007.md)
 - [InlineResponse2007Message](docs/InlineResponse2007Message.md)
 - [InlineResponse2007MessageBody](docs/InlineResponse2007MessageBody.md)
 - [InlineResponse2008](docs/InlineResponse2008.md)
 - [InlineResponse2008Message](docs/InlineResponse2008Message.md)
 - [InlineResponse2008MessageBody](docs/InlineResponse2008MessageBody.md)
 - [InlineResponse2009](docs/InlineResponse2009.md)
 - [InlineResponse2009Message](docs/InlineResponse2009Message.md)
 - [InlineResponse200Message](docs/InlineResponse200Message.md)
 - [InlineResponse200MessageBody](docs/InlineResponse200MessageBody.md)
 - [InlineResponse200MessageHeader](docs/InlineResponse200MessageHeader.md)
 - [Lyrics](docs/Lyrics.md)
 - [Snippet](docs/Snippet.md)
 - [Subtitle](docs/Subtitle.md)
 - [Track](docs/Track.md)
 - [TrackPrimaryGenres](docs/TrackPrimaryGenres.md)
 - [TrackPrimaryGenresMusicGenre](docs/TrackPrimaryGenresMusicGenre.md)
 - [TrackPrimaryGenresMusicGenreList](docs/TrackPrimaryGenresMusicGenreList.md)
 - [TrackSecondaryGenres](docs/TrackSecondaryGenres.md)
 - [TrackSecondaryGenresMusicGenre](docs/TrackSecondaryGenresMusicGenre.md)
 - [TrackSecondaryGenresMusicGenreList](docs/TrackSecondaryGenresMusicGenreList.md)


## Documentation For Authorization


## key

- **Type**: API key
- **API key parameter name**: apikey
- **Location**: URL query string


## Author

info@musixmatch.com

