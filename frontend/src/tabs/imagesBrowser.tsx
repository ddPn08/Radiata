import {
  Box,
  Container,
  Portal,
  Notification,
  Input,
  Flex,
  ActionIcon,
  NativeSelect,
} from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import { IconRotateClockwise } from '@tabler/icons-react'
import { GetAllImageFilesRequest } from 'modules/api/apis/MainApi'
import { useEffect, useState } from 'react'

import { api, createUrl } from '~/api'
import Gallery from '~/components/gallery'
import BetterNumInput from '~/components/ui/betterNumInput'
import { categoryList, categoryType } from '~/types/generate'
import { GeneratedImage } from '~/types/generatedImage'

const ImagesBrowser = () => {
  const [page, setPage] = useState<number>(0)
  const [pageLength, setPageLength] = useState<number>(0)
  const [category, setCategory] = useState<categoryType>('txt2img')
  const [images, setImages] = useState<GeneratedImage[]>([])
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const isLargeScreen = useMediaQuery('(min-width: 470px)', true)
  const count = 20

  const parseImages = (images: any, category: categoryType): GeneratedImage[] => {
    const data: GeneratedImage[] = []

    Object.entries(images).forEach(([key, value]: [string, any]) => {
      data.push({
        url: createUrl(`/api/images/${category}/${key}`),
        info: value.info,
      })
    })

    return data
  }

  const fetchImage = async (page: number, category: categoryType) => {
    try {
      const requestParam: GetAllImageFilesRequest = {
        category: category,
        page: page,
      }
      const res = await api.getAllImageFiles(requestParam)

      if (res.status !== 'success') {
        if (res.message) {
          setErrorMessage(res.message)
        } else {
          setErrorMessage('Something went wrong')
        }
      }
      setPageLength(Math.ceil(res.length / count))
      const data = parseImages(res.data, category)
      setImages(data)
    } catch (e) {
      console.error(e)
      setErrorMessage((e as Error).message)
    }
  }

  useEffect(() => {
    fetchImage(page, category)
  }, [page])
  useEffect(() => {
    fetchImage(page, category)
  }, [category])

  return (
    <>
      <Flex h={'100%'} direction="column">
        <Container py={'md'} w={'100%'}>
          <Flex gap={'sm'} direction={isLargeScreen ? 'row' : 'column'}>
            <Input.Wrapper label={'Category'}>
              <NativeSelect
                data={categoryList.map((e) => e)}
                value={category}
                w={'100%'}
                onChange={(e) => {
                  setCategory(e.currentTarget.value as categoryType)
                }}
              />
            </Input.Wrapper>

            <Input.Wrapper label="Page">
              <Flex w={'100%'} align={'center'} gap={'sm'} direction="row">
                <BetterNumInput
                  defaultValue={1}
                  value={page + 1}
                  min={1}
                  max={pageLength}
                  step={1}
                  onChange={(e) => e == undefined || setPage(e - 1)}
                  w={'100%'}
                />
                <ActionIcon
                  variant={'outline'}
                  color={'blue'}
                  onClick={() => fetchImage(page, category)}
                >
                  <IconRotateClockwise size={16} />
                </ActionIcon>
              </Flex>
            </Input.Wrapper>
          </Flex>
        </Container>
        <Box
          h="100%"
          w="100%"
          sx={{
            overflowY: 'auto',
          }}
        >
          <Container py={'md'}>
            <Gallery images={images} />
          </Container>
        </Box>
      </Flex>

      {errorMessage && (
        <Portal>
          <Notification
            title={'Error occured'}
            color={'red'}
            m={'md'}
            pos={'absolute'}
            bottom={0}
            right={0}
            onClose={() => {
              setErrorMessage(null)
            }}
          >
            {errorMessage}
          </Notification>
        </Portal>
      )}
    </>
  )
}

export default ImagesBrowser
