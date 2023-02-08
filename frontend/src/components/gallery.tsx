import { Box, Image, Portal, SimpleGrid, Skeleton } from '@mantine/core'
import { useState } from 'react'

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
            { maxWidth: 'sm', cols: 3 },
            { minWidth: 'sm', cols: 3 },
            { minWidth: 'md', cols: 3 },
            { minWidth: 'lg', cols: 4 },
            { minWidth: 'xl', cols: 5 },
          ]}
        >
          {images.map((image, i) => {
            return (
              <Box key={image.url}>
                <Image
                  src={image.url}
                  alt={image.info.prompt}
                  sx={{
                    cursor: 'pointer',
                  }}
                  onClick={() => {
                    setInitialIndex(i)
                    setShowOverlay(true)
                  }}
                />
              </Box>
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
