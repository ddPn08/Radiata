import {
  Button,
  Checkbox,
  Container,
  Input,
  NativeSelect,
  NumberInput,
  SimpleGrid,
  Space,
  Stack,
  Text,
  Progress,
  Box,
  Alert,
  Loader,
} from '@mantine/core'
import { useState } from 'react'
import SizeInput from '../components/sizeInput'
import { api } from '~/api'
import { useForm } from '@mantine/form'
import { BuildRequest } from 'internal:api'
import { IconInfoCircle } from '@tabler/icons-react'

interface BuildRequestForm extends Omit<BuildRequest, 'fp16'> {
  denoising_precision: string
}

const Engine = () => {
  const form = useForm<BuildRequestForm>({
    initialValues: {
      model_id: '',
      hf_token: '',
      denoising_precision: 'float16',
      opt_image_height: 512,
      opt_image_width: 512,
      max_batch_size: 1,
      onnx_opset: 16,
      build_static_batch: false,
      build_dynamic_shape: true,
      build_preview_features: false,
      force_engine_build: false,
      force_onnx_export: false,
      force_onnx_optimize: false,
      onnx_minimal_optimization: false,
    },
  })

  const [status, setStatus] = useState<Record<string, any> | null>(null)
  const [error, setError] = useState<string | null>(null)

  const onSubmit = async (req: BuildRequest) => {
    console.log(req)
    await buildEngine(req)
  }

  const buildEngine = async (req: BuildRequest) => {
    try {
      setError(null)
      setStatus({
        message: 'loading...',
        progress: 0,
      })
      const { raw } = await api.buildEngineRaw({ buildRequest: req })
      const reader = raw.body?.getReader()
      if (!reader) return
      let finish = false
      while (!finish) {
        const res = await reader.read()

        if (res.done) {
          finish = true
        }

        try {
          setStatus(JSON.parse(new TextDecoder().decode(res.value) || ''))
        } catch (_) {}
      }
      setStatus(null)
    } catch (e) {
      setStatus(null)
      setError((e as Error).message)
    }
  }

  return (
    <Container py={'md'}>
      <Text size={'lg'}>Build TensorRT from diffusers moodel on Hugging Face</Text>
      <form
        onSubmit={form.onSubmit((values) =>
          onSubmit({
            fp16: values.denoising_precision === 'float16',
            ...values,
          }),
        )}
      >
        <Stack my={'sm'}>
          <Input.Wrapper label={'Model ID'} withAsterisk>
            <Input
              placeholder="hugging face model id (e.g. CompVis/stablediffusion-v1-4)"
              {...form.getInputProps('model_id')}
            />
          </Input.Wrapper>

          <Input.Wrapper label={'Hugging Face Access Token'}>
            <Input placeholder="hf_********************" {...form.getInputProps('hf_token')} />
          </Input.Wrapper>

          <SizeInput
            label={'Optimization Image Width'}
            {...form.getInputProps('opt_image_width')}
          />

          <SizeInput
            label={'Optimization Image Height'}
            {...form.getInputProps('opt_image_height')}
          />

          <Input.Wrapper label={'Denoising precision'}>
            <NativeSelect data={['float32', 'float16']} {...form.getInputProps('fp16')} />
          </Input.Wrapper>

          <Input.Wrapper label={'Max batch size'}>
            <NumberInput
              min={1}
              max={32}
              defaultValue={1}
              {...form.getInputProps('max_batch_size')}
            />
          </Input.Wrapper>

          <SimpleGrid
            cols={4}
            spacing="lg"
            breakpoints={[
              { maxWidth: 'md', cols: 3, spacing: 'md' },
              { maxWidth: 'sm', cols: 2, spacing: 'sm' },
              { maxWidth: 'xs', cols: 1, spacing: 'sm' },
            ]}
          >
            <Checkbox label={'Build static batch'} {...form.getInputProps('build_static_path')} />
            <Checkbox
              label={'Build dynamic shape'}
              {...form.getInputProps('build_dynamic_shape')}
            />
            <Checkbox
              label={'Build preview features'}
              {...form.getInputProps('build_preview_features')}
            />
            <Checkbox label={'Force engine build'} {...form.getInputProps('force_engine_build')} />
            <Checkbox label={'Force onnx export'} {...form.getInputProps('force_onnx_export')} />
            <Checkbox
              label={'Force onnx optimize'}
              {...form.getInputProps('force_onnx_optimize')}
            />
            <Checkbox
              label={'Onnx minimal optimization'}
              {...form.getInputProps('onnx_minimal_optimization')}
            />
          </SimpleGrid>

          <Space h={'md'} />

          {status ? (
            <Box w={'100%'}>
              <Button w={'100%'} disabled>
                <Loader />
              </Button>
              <Alert title={'Status'}>
                <Text>{status['message']}</Text>
                <Progress value={status['progress'] * 100} />
              </Alert>
            </Box>
          ) : (
            <Button type={'submit'}>Build</Button>
          )}
        </Stack>
      </form>

      {error && (
        <Box>
          <Alert icon={<IconInfoCircle />} title={'Something went wrong...'} color={'red'}>
            {error}
          </Alert>
        </Box>
      )}
    </Container>
  )
}

export default Engine
