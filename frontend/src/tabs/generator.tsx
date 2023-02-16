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
import {
  GeneratedImage,
  GeneratorImageProgress,
  GeneratorImageResult,
} from '~/types/generatedImage'
import { streamGenerator } from '~/utils/stream'

const Generator = () => {
  const [parameters, setParameters] = useAtom(generationParametersAtom)
  const [loadingParameters, setLoadingParameters] = useState<GenerationParamertersForm>(parameters)
  const [images, setImages] = useState<GeneratedImage[]>([])
  const [performance, setPerformance] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [progress, setProgress] = useState<number | null>(null)

  const isLargeScreen = useMediaQuery('(min-width: 992px)', true)

  const parseImages = (
    images: string[],
    info: { [key: string]: string | number | boolean },
  ): GeneratedImage[] => {
    return Object.entries(images).map(([_, value]: [string, any]) => {
      return {
        url: createUrl(`/api/images/${info.img2img ? 'img2img' : 'txt2img'}/${value}`),
        info: info,
      }
    })
  }

  const onSubmit = async (values: GenerationParamertersForm) => {
    try {
      const requestBody: GenerationParameters = {
        ...values,
        scheduler_id: Scheduler[values.scheduler_id],
      }

      setIsLoading(true)
      setErrorMessage(null)
      setPerformance(null)
      setLoadingParameters(parameters)

      const { raw } = await api.generatorImageRaw({
        generateImageRequest: requestBody,
      })
      if (raw.body != null) {
        for await (const stream of streamGenerator(raw.body)) {
          if (stream.type == 'progress') {
            const data = stream as GeneratorImageProgress
            data.progress && setProgress(data.progress)
            data.performance && setPerformance(data.performance)
          } else if (stream.type == 'result') {
            const data = stream as GeneratorImageResult
            setImages((imgs) => [...parseImages(data.path, data.info), ...imgs])
            data.performance && setPerformance(data.performance)
            setProgress(null)
          }
        }
      }

      setIsLoading(false)
    } catch (e) {
      console.error(e)
      setErrorMessage((e as Error).message)
      setIsLoading(false)
    }
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
              disabled={isLoading}
              sx={{
                cursor: isLoading ? 'not-allowed' : 'pointer',
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
              <Gallery images={images} isLoading={isLoading} parameters={loadingParameters} />
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
