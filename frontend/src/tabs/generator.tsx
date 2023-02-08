import { Box, Button, Divider, Flex, Image, Stack, Textarea } from '@mantine/core'
import { useMediaQuery } from '@mantine/hooks'
import { useAtom } from 'jotai'
import { api } from '~/api'
import {
  GenerationParamertersForm,
  GenerationParameters,
  generationParametersAtom,
} from '~/atoms/generationParameters'
import Parameters from '~/components/parameters'
import { Scheduler } from '~/types/generate'

const Generator = () => {
  const [parameters, setParameters] = useAtom(generationParametersAtom)

  const isLargeScreen = useMediaQuery('(min-width: 992px)', true)

  const onModelChange = (value: string) => {}

  const onSubmit = async (values: GenerationParamertersForm) => {
    console.log(values)

    const requestBody: GenerationParameters = {
      ...values,
      scheduler_id: Scheduler[values.scheduler_id],
    }
    const res = await api.generateImage({
      generateImageRequest: requestBody,
    })

    console.log(res)

    // setResults(Object.entries(res.data.images) as any)
    // setTime(res.data.performance)
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
        <Flex h={'100%'} direction={isLargeScreen ? 'row' : 'column'}>
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
                autosize
              />
              <Textarea
                label={'Negative'}
                defaultValue={parameters.negative_prompt}
                onChange={(e) => setParameters((p) => ({ ...p, negative_prompt: e.target.value }))}
                autosize
              />
            </Stack>

            <Button type={'submit'}>Generate</Button>

            <Box h={isLargeScreen ? '80%' : '480px'}>
              <Image />
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
    </Box>
  )
}

export default Generator
