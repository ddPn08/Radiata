import { Carousel, Embla, useAnimationOffsetEffect } from '@mantine/carousel'
import { Box, Center, CloseButton, Flex, Table, Text, Drawer } from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import type { ImageInformation } from 'internal:api'
import { useState } from 'react'

interface Props {
  images: [string, ImageInformation][]
  initialIndex: number
  onClose: () => void
}

const imageUrl = (url: string) => {
  const regex = /^[A-Za-z0-9+/=]+$/
  return regex.test(url) ? `data:image/png;base64,${url}` : url
}

const OverlayPreview = ({ images, initialIndex, onClose }: Props) => {
  const [currentInfo, setCurrentInfo] = useState(images[initialIndex]![1])
  const [opened, setOpened] = useState(false)
  const [embla, setEmbla] = useState<Embla | null>(null)

  const TRANSITION_DURATION = 200
  useAnimationOffsetEffect(embla, TRANSITION_DURATION)
  const isLargeScreen = useMediaQuery('(min-width: 992px)', true)

  const onSlideChange = (index: number) => {
    setCurrentInfo(images[index]![1])
  }

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
      <Flex h={'100%'} w={'100%'}>
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
          onSlideChange={onSlideChange}
          getEmblaApi={setEmbla}
          loop
        >
          {images.map(([image]) => {
            return (
              <Carousel.Slide key={image}>
                <Center h={'100%'}>
                  <img
                    src={imageUrl(image)}
                    style={{
                      maxHeight: '100%',
                      maxWidth: '100%',
                      objectFit: 'contain',
                    }}
                    onClick={() => {
                      isLargeScreen || setOpened(true)
                    }}
                  />
                </Center>
              </Carousel.Slide>
            )
          })}
        </Carousel>
        <Box
          h={'100%'}
          w={'480px'}
          bg={'dark'}
          sx={{
            overflow: 'auto',
            display: isLargeScreen ? 'block' : 'none',
          }}
        >
          <Text size={'lg'} weight={'bold'} p={'md'}>
            Information
          </Text>
          <Table horizontalSpacing={'md'} verticalSpacing={'sm'} fontSize={'md'}>
            <thead>
              <tr>
                <th>Name</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(currentInfo).map(([key, value]) => {
                return (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>{value == null || String(value) == '' ? 'none' : String(value)}</td>
                  </tr>
                )
              })}
            </tbody>
          </Table>
        </Box>
      </Flex>
      <Drawer
        opened={opened}
        onClose={() => setOpened(false)}
        position={'right'}
        zIndex={1001}
        withCloseButton={false}
      >
        <Box
          h={'100svh'}
          sx={{
            overflow: 'auto',
          }}
        >
          <Text size={'lg'} weight={'bold'} p={'md'}>
            Information
          </Text>
          <Table horizontalSpacing={'md'} verticalSpacing={'sm'} fontSize={'md'}>
            <thead>
              <tr>
                <th>Name</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(currentInfo).map(([key, value]) => {
                return (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>{!value && String(value) == '' ? 'none' : String(value)}</td>
                  </tr>
                )
              })}
            </tbody>
          </Table>
        </Box>
      </Drawer>

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
