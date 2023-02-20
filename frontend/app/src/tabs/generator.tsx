import {
  Box,
  Button,
  Divider,
  Flex,
  Notification,
  Portal,
  Stack,
  Text,
  Textarea,
  Progress,
} from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import type {
  ImageGenerationError,
  ImageGenerationOptions,
  ImageGenerationProgress,
  ImageGenerationResult,
  ImageInformation,
} from 'internal:api'
import { useAtom } from 'jotai'
import { useState } from 'react'

import { api } from '~/api'
import { generationParametersAtom } from '~/atoms/generationParameters'
import Gallery from '~/components/gallery/gallery'
import Parameters from '~/components/parameters'
import { Scheduler } from '~/types/generate'
import { streamGenerator } from '~/utils/stream'

const Generator = () => {
  const [parameters, setParameters] = useAtom(generationParametersAtom)
  const [images, setImages] = useState<[string, ImageInformation][]>([])
  const [loadingParameters, setLoadingParameters] = useState<ImageGenerationOptions>(parameters)
  const [loadingCount, setLoadingCount] = useState<number>(0)
  const [performance, setPerformance] = useState<number | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [progress, setProgress] = useState<number | null>(null)

  const isLargeScreen = useMediaQuery('(min-width: 992px)', true)

  const onSubmit = async (values: ImageGenerationOptions) => {
    try {
      const requestBody: ImageGenerationOptions = {
        ...values,
        scheduler_id: Scheduler[values.scheduler_id as keyof typeof Scheduler],
      }

      setLoadingCount((parameters.batch_count ?? 1) * (parameters.batch_size ?? 1))
      setLoadingParameters(parameters)
      setErrorMessage(null)
      setPerformance(null)
      api.generatorImage
      const { raw } = await api.generatorImageRaw({
        imageGenerationOptions: requestBody,
      })
      if (raw.body != null) {
        for await (const stream of streamGenerator(raw.body)) {
          if (stream.type === 'progress') {
            const data = stream as ImageGenerationProgress
            setProgress(data.progress || null)
            setPerformance(data.performance)
          } else if (stream.type === 'result') {
            const data = stream as ImageGenerationResult
            setImages((prev) => [...Object.entries(data.images), ...prev])
            setLoadingCount((i) => i - Object.keys(data.images).length)
            data.performance && setPerformance(data.performance)
          } else if (stream.type === 'error') {
            const data = stream as ImageGenerationError
            throw new Error([data.error, data.message].filter((e) => e).join(': '))
          }
        }
      }
    } catch (e) {
      console.error(e)
      setErrorMessage((e as Error).message)
    }
    setProgress(null)
    setLoadingCount(0)
  }

  const onKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // if Enter + Ctrl or Enter + Cmd
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      onSubmit(parameters)
    }
  }

  return (
    <Box h={'100%'}>
      <form
        style={{
          height: '100%',
          overflow: isLargeScreen ? 'hidden' : 'auto',
        }}
        onSubmit={(e) => {
          e.preventDefault()
          onSubmit(parameters)
        }}
      >
        <Flex h={'100%'} w={'100%'} direction={isLargeScreen ? 'row' : 'column'}>
          <Stack w={'100%'} p={'md'}>
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
              disabled={loadingCount > 0}
              sx={{
                cursor: loadingCount > 0 ? 'not-allowed' : 'pointer',
              }}
            >
              <Text>{parameters.img ? 'Generate (img2img mode)' : 'Generate'}</Text>
            </Button>
            <Box mih="32px">
              {performance && <Text align="end">Time: {performance.toFixed(2)}s</Text>}
              {progress && <Progress sections={[{ value: progress * 100, color: 'blue' }]} />}
            </Box>
            <Box
              mah={isLargeScreen ? '80%' : '480px'}
              pos={'relative'}
              sx={{
                overflowY: 'auto',
              }}
            >
              <Gallery
                images={images}
                loadingCount={loadingCount}
                ratio={loadingParameters.image_width! / loadingParameters.image_height!}
              />
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
            sx={{
              overflow: isLargeScreen ? 'auto' : 'visible',
            }}
          >
            <Parameters />
          </Box>
        </Flex>
      </form>

      {errorMessage && (
        <Portal>
          <Notification
            title={'Error occured'}
            sx={{
              'z-index': 1000000,
            }}
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
    </Box>
  )
}

export default Generator
