# Sample Images

This folder contains local test files for checking the upload flow.

## Valid files for quick testing

- `test_skin_light.jpg`
- `test_skin_medium.jpg`
- `test_skin_dark.jpg`

These are synthetic placeholder images created only to verify that:

- the app accepts readable JPG files
- the upload route works
- the prediction or inconclusive flow renders correctly

They are not medical samples and should not be used to judge model quality.

## Invalid files from earlier download attempts

Files moved into `invalid_downloads/` are HTML pages returned by stale DermNet URLs. They are not real images and should not be uploaded to the app.
