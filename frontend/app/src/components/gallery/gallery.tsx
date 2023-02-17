import { Box, Portal, SimpleGrid, Skeleton, AspectRatio } from '@mantine/core'
import type { ImageGenerationOptions, ImageInformation } from 'internal:api'
import { useState } from 'react'

import GalleryImage from './galleryImage'
import OverlayPreview from './overlayPreview'

interface Props {
  images: [string, ImageInformation][]
  isLoading: boolean
  parameters: ImageGenerationOptions | null
}

const Gallery = ({ images, isLoading, parameters }: Props) => {
  const [showOverlay, setShowOverlay] = useState(false)
  const [initialIndex, setInitialIndex] = useState(0)

  return (
    <>
      <Box>
        <SimpleGrid
          cols={3}
          breakpoints={[
            { maxWidth: 'xs', cols: 2 },
            { maxWidth: 'sm', cols: 2 },
            { minWidth: 'sm', cols: 2 },
            { minWidth: 'md', cols: 2 },
            { minWidth: 'lg', cols: 3 },
            { minWidth: 'xl', cols: 4 },
          ]}
        >
          {isLoading &&
            parameters != null &&
            [...Array(parameters.batch_count)].map((_, key) => {
              return (
                <AspectRatio
                  key={key}
                  ratio={(parameters.image_width || 512) / (parameters.image_height || 768)}
                >
                  <Skeleton />
                </AspectRatio>
              )
            })}
          {images.map(([image, info], i) => {
            return (
              <GalleryImage
                key={i}
                image={image}
                info={info}
                onClick={() => {
                  setInitialIndex(i)
                  setShowOverlay(true)
                }}
              />
            )
          })}
        </SimpleGrid>
      </Box>

      {showOverlay && (
        <Portal>
          <OverlayPreview
            images={images}
            initialIndex={initialIndex}
            onClose={() => {
              setShowOverlay(false)
            }}
          />
        </Portal>
      )}
    </>
  )
}

export default Gallery
