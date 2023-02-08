import { Carousel } from '@mantine/carousel'
import { Box, Center, CloseButton, Flex } from '@mantine/core'

import { GeneratedImage } from '~/types/generatedImage'

interface Props {
  images: GeneratedImage[]
  initialIndex: number
  onClose: () => void
}

const OverlayPreview = ({ images, initialIndex, onClose }: Props) => {
  return (
    <Box
      pos={'absolute'}
      top={0}
      left={0}
      w={'100%'}
      h={'100%'}
      opacity={1}
      sx={{
        zIndex: 1000,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        backdropFilter: 'blur(2px)',
        overflow: 'hidden',
      }}
      tabIndex={0}
      onLoad={(e) => {
        e.currentTarget.focus()
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          onClose()
        }
      }}
    >
      <Flex h={'100vh'} w={'100%'}>
        <Carousel
          w={'100%'}
          my={'auto'}
          slideSize={'80%'}
          slideGap={'md'}
          initialSlide={initialIndex}
          withIndicators
          sx={{
            overflowY: 'hidden',
          }}
          loop
        >
          {images.map((image) => {
            return (
              <Carousel.Slide key={image.url}>
                <Center h={'100%'}>
                  <img
                    src={image.url}
                    style={{
                      maxHeight: '100%',
                      maxWidth: '100%',
                      objectFit: 'contain',
                    }}
                  />
                </Center>
              </Carousel.Slide>
            )
          })}
        </Carousel>
        <Box h={'100%'} w={'400px'} bg={'dark'}></Box>
      </Flex>
      <CloseButton
        variant={'filled'}
        title={'Close previews'}
        iconSize={16}
        pos={'absolute'}
        top={0}
        left={0}
        m={'md'}
        onClick={onClose}
      />
    </Box>
  )
}

export default OverlayPreview
