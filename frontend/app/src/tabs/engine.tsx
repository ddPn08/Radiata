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
  Box,
  Alert,
  Loader,
} from '@mantine/core'
import { IconInfoCircle } from '@tabler/icons-react'
import type { BuildEngineOptions } from 'internal:api'
import { useAtom } from 'jotai'
import { useState } from 'react'

import NumberSliderInput from '../components/ui/numberSliderInput'

import { api } from '~/api'
import { engineFormAtom } from '~/atoms/engine'
import { IMAGE_SIZE_STEP, MAX_IMAGE_SIZE, MIN_IMAGE_SIZE } from '~/utils/static'

const Engine = () => {
  const [form, setForm] = useAtom(engineFormAtom)

  const [status, setStatus] = useState<Record<string, any> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<boolean | null>(null)

  const onSubmit = () => buildEngine(form)

  const buildEngine = async (req: BuildEngineOptions) => {
    try {
      setError(null)
      setSuccess(null)
      setStatus({
        message: 'loading...',
        progress: 0,
      })
      const { raw } = await api.buildEngineRaw({ buildEngineOptions: req })
      const reader = raw.body?.getReader()
      if (!reader) return
      let finish = false
      while (!finish) {
        const res = await reader.read()

        if (res.done) {
          setSuccess(true)
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
      setSuccess(false)
    }
  }

  return (
    <Box
      h="100%"
      sx={{
        overflowY: 'auto',
      }}
    >
      <Container py={'md'}>
        <Text size={'lg'}>Build TensorRT from diffusers moodel on Hugging Face</Text>
        <form
          onSubmit={(e) => {
            e.preventDefault()
            onSubmit()
          }}
        >
          <Stack my={'sm'}>
            <Input.Wrapper label={'Model ID (required)'} withAsterisk>
              <Input
                placeholder="hugging face model id (e.g. CompVis/stablediffusion-v1-4)"
                defaultValue={form.model_id}
                onChange={(e) => setForm({ ...form, model_id: e.currentTarget.value })}
              />
            </Input.Wrapper>

            <Input.Wrapper label={'Hugging Face Access Token'}>
              <Input
                placeholder="hf_********************"
                defaultValue={form.hf_token}
                onChange={(e) =>
                  setForm({
                    ...form,
                    hf_token: e.currentTarget.value,
                  })
                }
              />
            </Input.Wrapper>

            <NumberSliderInput
              label={'Optimization Image Width'}
              defaultValue={form.opt_image_width}
              min={MIN_IMAGE_SIZE}
              max={MAX_IMAGE_SIZE}
              step={IMAGE_SIZE_STEP}
              onChange={(value) => setForm({ ...form, opt_image_width: value })}
            />

            <NumberSliderInput
              label={'Optimization Image Height'}
              defaultValue={form.opt_image_height}
              min={MIN_IMAGE_SIZE}
              max={MAX_IMAGE_SIZE}
              step={IMAGE_SIZE_STEP}
              onChange={(value) => setForm({ ...form, opt_image_height: value })}
            />

            <Input.Wrapper label={'Denoising precision'}>
              <NativeSelect
                data={['float32', 'float16']}
                defaultValue={form.fp16 ? 'float16' : 'float32'}
                onChange={(e) => setForm({ ...form, fp16: e.currentTarget.value === 'float16' })}
              />
            </Input.Wrapper>

            <Input.Wrapper label={'Max batch size'}>
              <NumberInput
                min={1}
                max={32}
                defaultValue={form.max_batch_size}
                onChange={(value) => setForm({ ...form, max_batch_size: value || 1 })}
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
              <Checkbox
                label={'Build static batch'}
                defaultChecked={form.build_static_batch}
                onChange={(e) => setForm({ ...form, build_static_batch: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Build dynamic shape'}
                defaultChecked={form.build_dynamic_shape}
                onChange={(e) => setForm({ ...form, build_dynamic_shape: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Build preview features'}
                defaultChecked={form.build_preview_features}
                onChange={(e) =>
                  setForm({ ...form, build_preview_features: e.currentTarget.checked })
                }
              />
              <Checkbox
                label={'Force engine build'}
                defaultChecked={form.force_engine_build}
                onChange={(e) => setForm({ ...form, force_engine_build: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Force onnx export'}
                defaultChecked={form.force_onnx_export}
                onChange={(e) => setForm({ ...form, force_onnx_export: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Force onnx optimize'}
                defaultChecked={form.force_onnx_optimize}
                onChange={(e) => setForm({ ...form, force_onnx_optimize: e.currentTarget.checked })}
              />
              <Checkbox
                label={'Onnx minimal optimization'}
                defaultChecked={form.onnx_minimal_optimization}
                onChange={(e) =>
                  setForm({ ...form, onnx_minimal_optimization: e.currentTarget.checked })
                }
              />
            </SimpleGrid>

            {form.build_dynamic_shape && (
              <>
                <Input.Wrapper label={'Min latent resolution'}>
                  <NumberInput
                    defaultValue={form.min_latent_resolution}
                    onChange={(value) => setForm({ ...form, min_latent_resolution: value || 256 })}
                  />
                </Input.Wrapper>
                <Input.Wrapper label={'Max latent resolution'}>
                  <NumberInput
                    defaultValue={form.max_latent_resolution}
                    onChange={(value) => setForm({ ...form, max_latent_resolution: value || 1024 })}
                  />
                </Input.Wrapper>
              </>
            )}

            <Space h={'md'} />

            {status ? (
              <Box w={'100%'}>
                <Alert title={'Processing...'}>
                  <Text>
                    This may take about 10 minutes. Please wait until the process is finished.
                  </Text>
                </Alert>
                <Button w={'100%'} my={'sm'} disabled>
                  <Loader p={'xs'} />
                </Button>
              </Box>
            ) : (
              <Button type={'submit'}>Build</Button>
            )}

            {success && (
              <Box>
                <Alert
                  title={'Success!'}
                  color={'green'}
                  withCloseButton={true}
                  onClose={() => {
                    setSuccess(null)
                  }}
                >
                  <Text>The model has been built successfully. You can now generate images!</Text>
                </Alert>
              </Box>
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
    </Box>
  )
}

export default Engine
