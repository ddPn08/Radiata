import { Box, Image, SimpleGrid, Skeleton } from '@mantine/core'
import { GeneratedImage } from '~/types/generatedImage'

interface Props {
  images: GeneratedImage[]
  isLoading: boolean
}

const Gallery = ({ images, isLoading }: Props) => {
  return (
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
        breakpoints={[
          { maxWidth: 'xs', cols: 1 },
          { maxWidth: 'sm', cols: 2 },
          { minWidth: 'sm', cols: 2 },
          { minWidth: 'md', cols: 2 },
          { minWidth: 'lg', cols: 3 },
          { minWidth: 'xl', cols: 4 },
        ]}
      >
        {images.map((image) => {
          return (
            <Box key={image.url}>
              <Image src={image.url} />
            </Box>
          )
        })}
      </SimpleGrid>
    </Box>
  )
}

export default Gallery
