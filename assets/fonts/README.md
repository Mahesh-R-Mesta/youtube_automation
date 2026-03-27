# Fonts

Place your `.ttf` font files here. The pipeline will automatically discover them.

## Required

- **OpenSans-Bold.ttf** — used for video subtitles and YouTube thumbnails.

## Download

Download from Google Fonts (free, OFL license):

```
https://fonts.google.com/specimen/Open+Sans
```

Direct download link (select **Bold 700**):
1. Go to https://fonts.google.com/specimen/Open+Sans
2. Click "Download family"
3. Extract the zip → find `static/OpenSans-Bold.ttf`
4. Copy it to this directory: `assets/fonts/OpenSans-Bold.ttf`

## Fallback

If no font is found here, the pipeline falls back to Windows system fonts:
- `C:\Windows\Fonts\arialbd.ttf` (Arial Bold)
- `C:\Windows\Fonts\arial.ttf` (Arial Regular)

A system font will always be present on Windows, so the pipeline will work even without this file.
