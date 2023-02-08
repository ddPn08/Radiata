import { Box, Button, Divider, Flex, Notification, Stack, Textarea } from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import { useAtom } from 'jotai'
import { useState } from 'react'
import { api, createUrl } from '~/api'
import {
  GenerationParamertersForm,
  GenerationParameters,
  generationParametersAtom,
} from '~/atoms/generationParameters'
import Gallery from '~/components/gallery'
import Parameters from '~/components/parameters'
import { Scheduler } from '~/types/generate'
import { GeneratedImage } from '~/types/generatedImage'

const Generator = () => {
  const [parameters, setParameters] = useAtom(generationParametersAtom)
  const [images, setImages] = useState<GeneratedImage[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const isLargeScreen = useMediaQuery('(min-width: 992px)', true)

  const parseImages = (images: any): GeneratedImage[] => {
    const data: GeneratedImage[] = []

    Object.entries(images).forEach(([key, value]: [string, any]) => {
      data.push({
        url: createUrl(`/api/images/${value.info.img2img ? 'img2img' : 'txt2img'}/${key}`),
        info: value.info,
      })
    })

    return data
  }

  const onSubmit = async (values: GenerationParamertersForm) => {
    console.log(values)

    const requestBody: GenerationParameters = {
      ...values,
      scheduler_id: Scheduler[values.scheduler_id],
    }

    setIsLoading(true)
    const res = await api.generateImage({
      generateImageRequest: requestBody,
    })
    setIsLoading(false)

    console.log(res.data)

    if (res.status !== 'success') {
      if (res.message) {
        setErrorMessage(res.message)
      } else {
        setErrorMessage('Something went wrong')
      }
    }

    const data = parseImages(res.data.images)
    setImages((imgs) => [...data, ...imgs])
  }

  const onKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // if Enter + Ctrl or Enter + Cmd
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      onSubmit(parameters)
    }
  }

  return (
    <Box
      h={'100%'}
      sx={{
        overflow: isLargeScreen ? 'hidden' : 'scroll',
      }}
    >
      <form
        style={{
          height: '100%',
        }}
        onSubmit={(e) => {
          e.preventDefault()
          onSubmit(parameters)
        }}
      >
        <Flex h={'100%'} w={'100%'} direction={isLargeScreen ? 'row' : 'column'}>
          <Stack
            w={'100%'}
            p={'md'}
            sx={{
              overflow: 'hidden',
            }}
          >
            <Stack w={'100%'}>
              <Textarea
                label={'Positive'}
                defaultValue={parameters.prompt}
                onChange={(e) => {
                  setParameters((p) => ({
                    ...p,
                    prompt: e.target.value,
                  }))
                }}
                onKeyDown={onKeyPress}
                autosize
              />
              <Textarea
                label={'Negative'}
                defaultValue={parameters.negative_prompt}
                onChange={(e) => setParameters((p) => ({ ...p, negative_prompt: e.target.value }))}
                onKeyDown={onKeyPress}
                autosize
              />
            </Stack>

            <Button
              mih={'36px'}
              type={'submit'}
              disabled={isLoading}
              sx={{
                cursor: isLoading ? 'not-allowed' : 'pointer',
              }}
            >
              Generate
            </Button>

            <Box
              h={isLargeScreen ? '80%' : '480px'}
              sx={{
                overflow: 'scroll',
              }}
            >
              <Gallery images={images} isLoading={isLoading} />
            </Box>
          </Stack>

          <Divider orientation={isLargeScreen ? 'vertical' : 'horizontal'} />

          <Box
            w={
              isLargeScreen
                ? {
                    md: 640,
                    lg: 720,
                  }
                : '100%'
            }
          >
            <Parameters />
          </Box>
        </Flex>
      </form>

      {errorMessage && (
        <Notification
          title={'Error occured'}
          color={'red'}
          onClose={() => {
            setErrorMessage(null)
          }}
        >
          {errorMessage}
        </Notification>
      )}
    </Box>
  )
}

export default Generator
