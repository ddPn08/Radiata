import { Box, Portal, SimpleGrid, Skeleton } from '@mantine/core'
import { useState } from 'react'

import GalleryImage from './galleryImage'
import OverlayPreview from './overlayPreview'

import { GeneratedImage } from '~/types/generatedImage'

interface Props {
  images: GeneratedImage[]
  isLoading: boolean
}

const Gallery = ({ images, isLoading }: Props) => {
  const [showOverlay, setShowOverlay] = useState(false)
  const [initialIndex, setInitialIndex] = useState(0)

  return (
    <>
      <Box>
        {isLoading && (
          <Skeleton
            h={{
              xs: 300,
              sm: 400,
              md: 300,
            }}
            my={'sm'}
          />
        )}
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
          {images.map((image, i) => {
            return (
              <GalleryImage
                key={image.url}
                image={image}
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
