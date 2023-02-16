import { Box, Portal, SimpleGrid, Skeleton, AspectRatio } from '@mantine/core'
import { useState } from 'react'

import GalleryImage from './galleryImage'
import OverlayPreview from './overlayPreview'

import { GeneratedImage } from '~/types/generatedImage'

interface Props {
  images: GeneratedImage[]
  loadingCount?: number | null
  ratio?: number | null
}

const Gallery = ({ images, loadingCount, ratio }: Props) => {
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
          {ratio &&
            [...Array(loadingCount ?? 0)].map((_, key) => {
              return (
                <AspectRatio key={key} ratio={ratio}>
                  <Skeleton />
                </AspectRatio>
              )
            })}
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
